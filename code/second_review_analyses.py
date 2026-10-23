"""Controlled component comparisons and verification sensitivities.

All machine-readable outputs are written to ``results``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genpareto
from sklearn.linear_model import LogisticRegression

import evaluation as ev
from data import load_drivers, load_real
from major_revision_analysis import (city_paths, seasonal_event_prob,
                                     fit_seasonal_gpd)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
U = 1.0
OCC_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TAIL_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Exact threshold-weighted CRPS for a deterministic hurdle-GPD forecast
def _support_cap(y, sigma, xi):
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    if xi < 0:
        return np.minimum(y, -sigma / xi)
    return y


def gpd_survival_integral(y, sigma, xi):
    """Integral of the GPD survival function from zero to y."""
    y = _support_cap(np.maximum(y, 0.0), sigma, xi)
    sigma = np.asarray(sigma, float)
    if abs(xi) < 1e-7:
        return sigma * (1.0 - np.exp(-y / sigma))
    base = np.maximum(1.0 + xi * y / sigma, 0.0)
    if abs(xi - 1.0) < 1e-7:
        return sigma * np.log(np.maximum(base, 1e-300))
    return sigma / (xi - 1.0) * (base ** (1.0 - 1.0 / xi) - 1.0)


def gpd_survival_sq_tail(a, sigma, xi):
    """Integral from a to the endpoint/infinity of squared GPD survival."""
    sigma = np.asarray(sigma, float)
    a = np.maximum(np.asarray(a, float), 0.0)
    if xi < 0:
        endpoint = -sigma / xi
        outside = a >= endpoint
    else:
        outside = np.zeros(np.broadcast(a, sigma).shape, dtype=bool)
    if abs(xi) < 1e-7:
        out = 0.5 * sigma * np.exp(-2.0 * a / sigma)
    else:
        base = np.maximum(1.0 + xi * a / sigma, 0.0)
        out = sigma / (2.0 - xi) * base ** (1.0 - 2.0 / xi)
    return np.where(outside, 0.0, out)


def exact_twcrps_parts(index, pi, sigma, xi, tau=1.0, u=1.0):
    """Exact score and its observation, reward and penalty terms (tau >= u)."""
    index = np.asarray(index, float)
    pi = np.asarray(pi, float)
    sigma = np.asarray(sigma, float)
    a = float(tau - u)
    y = np.maximum(index - u, 0.0)
    event = (y > a).astype(float)
    ycap = np.maximum(y, a)
    int_a_y = (gpd_survival_integral(ycap, sigma, xi)
               - gpd_survival_integral(np.full_like(ycap, a), sigma, xi))
    observation = event * (y - a)
    reward = -2.0 * event * pi * int_a_y
    penalty = pi ** 2 * gpd_survival_sq_tail(a, sigma, xi)
    return observation + reward + penalty, observation, reward, penalty


# ---------------------------------------------------------------------------
# Explicitly penalised linear hurdle with unpenalised intercepts and xi
class TailRidge(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        self.xi_raw = nn.Parameter(torch.tensor(0.0))

    def params(self, x):
        sigma = F.softplus(self.linear(x).squeeze(-1)) + 1e-4
        xi = 0.5 * torch.tanh(self.xi_raw)
        return sigma, xi

    def loss(self, x, y, lam):
        sigma, xi = self.params(x)
        base = 1.0 + xi * y / sigma
        xi_safe = torch.where(xi.abs() < 1e-4,
                              torch.full_like(xi, 1e-4), xi)
        generic = torch.where(
            base > 0,
            torch.log(sigma) + (1.0 / xi_safe + 1.0) * torch.log(base.clamp_min(1e-8)),
            torch.full_like(base, 1e4),
        )
        exponential = torch.log(sigma) + y / sigma
        nll = torch.where(xi.abs() < 1e-4, exponential, generic).mean()
        return nll + 0.5 * lam * (self.linear.weight ** 2).sum()


def _standardise(x_train, *others):
    mu = x_train.mean(0)
    sd = x_train.std(0) + 1e-8
    return [(x_train - mu) / sd] + [(x - mu) / sd for x in others]


def _fit_tail(x, y, lam, epochs=1200, seed=0):
    torch.manual_seed(seed)
    model = TailRidge(x.shape[1])
    # Initialise the scale near the stationary estimate.
    xi0, _, sig0 = genpareto.fit(y.numpy(), floc=0.0)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(float(np.log(np.expm1(max(sig0, 1e-3)))))
        model.xi_raw.fill_(float(np.arctanh(np.clip(xi0 / 0.5, -0.95, 0.95))))
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    best, stale = np.inf, 0
    for _ in range(epochs):
        opt.zero_grad()
        loss = model.loss(x, y, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        value = float(loss.detach())
        if value < best - 1e-7:
            best, stale = value, 0
        else:
            stale += 1
        if stale >= 150:
            break
    return model


def _tail_nll(model, x, y):
    with torch.no_grad():
        sigma, xi = model.params(x)
        base = 1.0 + xi * y / sigma
        xi_safe = torch.where(xi.abs() < 1e-4,
                              torch.full_like(xi, 1e-4), xi)
        generic = torch.where(
            base > 0,
            torch.log(sigma) + (1.0 / xi_safe + 1.0) * torch.log(base.clamp_min(1e-8)),
            torch.full_like(base, 1e4),
        )
        exponential = torch.log(sigma) + y / sigma
        return float(torch.where(xi.abs() < 1e-4, exponential, generic).mean())


def fit_penalised_hurdle_cv(x_train_raw, o_train, excess_train, x_test_raw):
    n = len(x_train_raw)
    folds = [(0, int(.55*n), int(.70*n)),
             (0, int(.70*n), int(.85*n)),
             (0, int(.85*n), n)]
    occ_losses = {lam: [] for lam in OCC_LAMBDAS}
    tail_losses = {lam: [] for lam in TAIL_LAMBDAS}
    for fold_no, (start, train_end, valid_end) in enumerate(folds):
        tr = np.arange(start, train_end)
        va = np.arange(train_end, valid_end)
        xtr, xva = _standardise(x_train_raw[tr], x_train_raw[va])
        ytr_o = o_train[tr].numpy()
        yva_o = o_train[va].numpy()
        for lam in OCC_LAMBDAS:
            lr = LogisticRegression(C=1.0/lam, solver="lbfgs",
                                    max_iter=3000)
            lr.fit(xtr.numpy(), ytr_o)
            p = np.clip(lr.predict_proba(xva.numpy())[:, 1], 1e-8, 1-1e-8)
            occ_losses[lam].append(float(-np.mean(yva_o*np.log(p)
                                                 +(1-yva_o)*np.log(1-p))))
        em_tr = o_train[tr] > 0.5
        em_va = o_train[va] > 0.5
        for lam in TAIL_LAMBDAS:
            tm = _fit_tail(xtr[em_tr], excess_train[tr][em_tr], lam,
                           seed=100*fold_no + int(-np.log10(lam)))
            tail_losses[lam].append(_tail_nll(tm, xva[em_va],
                                               excess_train[va][em_va]))
    lam_occ = min(OCC_LAMBDAS, key=lambda x: np.mean(occ_losses[x]))
    lam_tail = min(TAIL_LAMBDAS, key=lambda x: np.mean(tail_losses[x]))
    xtr, xte = _standardise(x_train_raw, x_test_raw)
    lr = LogisticRegression(C=1.0/lam_occ, solver="lbfgs",
                            max_iter=5000)
    lr.fit(xtr.numpy(), o_train.numpy())
    pi = lr.predict_proba(xte.numpy())[:, 1]
    ext = o_train > 0.5
    tail = _fit_tail(xtr[ext], excess_train[ext], lam_tail, seed=2026)
    with torch.no_grad():
        sigma, xi = tail.params(xte)
    return dict(pi=pi, sigma=sigma.numpy(), xi=float(xi),
                lambda_occ=lam_occ, lambda_tail=lam_tail,
                occurrence_model=lr, tail_model=tail)


# ---------------------------------------------------------------------------
def city_driver_groups(city, extra_lag):
    p = city_paths(city)
    tele = load_drivers(p["tele"], lag_monthly_teleconnections=True)
    era = load_drivers(p["era"], extra_era5_lag_days=extra_lag)
    conv = load_drivers(p["conv"], extra_era5_lag_days=extra_lag)
    return tele, era, conv


def load_with_drivers(city, drivers):
    p = city_paths(city)
    data, dates = load_real(p["index"], n_lags=14, u=U, horizon=1,
                            drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    return data, dates, cut


def seasonal_parameters(data, dates, cut):
    itr = data["I"][:cut].numpy()
    otr = data["O"][:cut].numpy()
    p = seasonal_event_prob(otr, dates[:cut], dates[cut:])
    theta, ok = fit_seasonal_gpd(itr, dates[:cut])
    beta, xi = theta[:3], 0.5*np.tanh(theta[3])
    doy = dates[cut:].dayofyear.to_numpy()
    design = np.column_stack([np.ones(len(doy)), np.sin(2*np.pi*doy/365.25),
                              np.cos(2*np.pi*doy/365.25)])
    sigma = np.exp(np.clip(design @ beta, -8, 8))
    pot = ev.fit_pot(itr, U)
    return dict(pi=p, sigma=sigma, xi=float(xi), pot=pot, optimiser_ok=ok)


def moving_block_indices(n, rng, block):
    starts = rng.integers(0, max(1, n-block+1), size=int(np.ceil(n/block)))
    return np.concatenate([np.arange(s, min(s+block, n)) for s in starts])[:n]


def paired_interval(new, old, denominator, block, seed=20261022):
    new, old, denominator = map(np.asarray, (new, old, denominator))
    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = moving_block_indices(len(new), rng, block)
        vals[b] = (old[ii].mean()-new[ii].mean())/denominator[ii].mean()
    return np.quantile(vals, [.025, .975])


def conditional_calendar_interval(new_ext, old_ext, event_mask, block,
                                  seed=20261022):
    full_new = np.full(len(event_mask), np.nan)
    full_old = np.full(len(event_mask), np.nan)
    full_new[event_mask] = new_ext
    full_old[event_mask] = old_ext
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        ii = moving_block_indices(len(event_mask), rng, block)
        keep = event_mask[ii]
        if keep.sum() > 1:
            vals.append(np.nanmean(full_old[ii][keep]-full_new[ii][keep]))
    return np.quantile(vals, [.025, .975])


def reliability_terms(y, p, n_bins=10):
    out = ev.brier_decomposition(y, p, n_bins=n_bins)
    p = np.clip(np.asarray(p), 1e-12, 1-1e-12)
    out["LogS"] = float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    return out


def analyse_city(city):
    print(f"\n{'='*72}\nSECOND-REVIEW ANALYSES: {city.upper()}\n{'='*72}", flush=True)
    tele2, era2, conv2 = city_driver_groups(city, extra_lag=1)
    full2 = pd.concat([tele2, era2, conv2], axis=1)
    full2 = full2.loc[:, ~full2.columns.duplicated()]
    data, dates, cut = load_with_drivers(city, full2)
    index_test = data["I"][cut:].numpy()
    event_test = data["O"][cut:].numpy()
    seasonal = seasonal_parameters(data, dates, cut)
    seasonal_score, *seasonal_parts = exact_twcrps_parts(
        index_test, seasonal["pi"], seasonal["sigma"], seasonal["xi"])

    print("Fitting properly penalised full linear hurdle (lag-2 ERA5)", flush=True)
    linear = fit_penalised_hurdle_cv(data["X"][:cut], data["O"][:cut],
                                     data["excess"][:cut], data["X"][cut:])
    linear_score, *linear_parts = exact_twcrps_parts(
        index_test, linear["pi"], linear["sigma"], linear["xi"])

    # Harmonics-only and history-only stages separate calendar seasonality from persistence.
    base, _, base_cut = load_with_drivers(city, None)
    harmonics_cols = [18, 19]
    history_cols = list(range(18))
    print("Fitting harmonics-only and history-only stages", flush=True)
    harmonics = fit_penalised_hurdle_cv(
        base["X"][:base_cut, harmonics_cols], base["O"][:base_cut],
        base["excess"][:base_cut], base["X"][base_cut:, harmonics_cols])
    history = fit_penalised_hurdle_cv(
        base["X"][:base_cut, history_cols], base["O"][:base_cut],
        base["excess"][:base_cut], base["X"][base_cut:, history_cols])
    index_hist = fit_penalised_hurdle_cv(
        base["X"][:base_cut], base["O"][:base_cut],
        base["excess"][:base_cut], base["X"][base_cut:])

    stage_rows = []
    for name, fit in (("harmonics only", harmonics),
                      ("history without harmonics", history),
                      ("history plus harmonics", index_hist),
                      ("full lag-2 atmospheric", linear)):
        bs = (fit["pi"]-event_test)**2
        tw, *_ = exact_twcrps_parts(index_test, fit["pi"], fit["sigma"], fit["xi"])
        stage_rows.append(dict(
            city=city, model=name, n_predictors=(2 if name == "harmonics only"
                else 18 if name == "history without harmonics"
                else 20 if name == "history plus harmonics" else data["X"].shape[1]),
            BSS_vs_seasonal=1-bs.mean()/np.mean((seasonal["pi"]-event_test)**2),
            twCRPSS_vs_seasonal=1-tw.mean()/seasonal_score.mean(),
            lambda_occ=fit["lambda_occ"], lambda_tail=fit["lambda_tail"],
            xi=fit["xi"]))
    pd.DataFrame(stage_rows).to_csv(OUT/f"{city}_penalised_stages.csv", index=False)

    # Timing sensitivity: identical model and feature set, daily ERA5 at t-1 vs t-2.
    print("Fitting lag-1 daily-ERA5 sensitivity model", flush=True)
    tele1, era1, conv1 = city_driver_groups(city, extra_lag=0)
    full1 = pd.concat([tele1, era1, conv1], axis=1)
    full1 = full1.loc[:, ~full1.columns.duplicated()]
    data1, dates1, cut1 = load_with_drivers(city, full1)
    linear1 = fit_penalised_hurdle_cv(data1["X"][:cut1], data1["O"][:cut1],
                                      data1["excess"][:cut1], data1["X"][cut1:])
    timing_rows = []
    for lag, fit, dat, c in ((2, linear, data, cut), (1, linear1, data1, cut1)):
        ite = dat["I"][c:].numpy(); ote = dat["O"][c:].numpy()
        ss = seasonal_score
        score, *_ = exact_twcrps_parts(ite, fit["pi"], fit["sigma"], fit["xi"])
        timing_rows.append(dict(city=city, era5_latest_day=f"t-{lag}",
                                BSS_vs_seasonal=1-np.mean((fit["pi"]-ote)**2)
                                /np.mean((seasonal["pi"]-ote)**2),
                                twCRPSS_vs_seasonal=1-score.mean()/ss.mean(),
                                lambda_occ=fit["lambda_occ"],
                                lambda_tail=fit["lambda_tail"], xi=fit["xi"]))
    pd.DataFrame(timing_rows).to_csv(OUT/f"{city}_timing_sensitivity.csv", index=False)

    # Load the five revised GP fits produced by gp_review_sensitivity.py.
    gpfile = OUT/f"{city}_gp_main_predictions.npz"
    if not gpfile.exists():
        raise FileNotFoundError(f"Run gp_review_sensitivity.py first: {gpfile}")
    gp = np.load(gpfile)
    gp_pi, gp_sigma, gp_xi = gp["pi"], gp["sigma"], gp["xi"]
    gp_scores, gp_parts = [], []
    for s in range(len(gp_xi)):
        score, obs, reward, penalty = exact_twcrps_parts(
            index_test, gp_pi[s], gp_sigma[s], float(gp_xi[s]))
        gp_scores.append(score); gp_parts.append((obs, reward, penalty))
    gp_score = np.mean(gp_scores, axis=0)
    gp_pi_mean = gp_pi.mean(0)

    # Factorial component swaps.  Components are fitted once and recombined.
    pot_pi, pot_sigma, pot_xi = seasonal["pot"]
    stationary_sigma = np.full(len(index_test), pot_sigma)
    factorial_specs = {
        "seasonal occurrence + seasonal intensity":
            [(seasonal["pi"], seasonal["sigma"], seasonal["xi"])],
        "linear occurrence + seasonal intensity":
            [(linear["pi"], seasonal["sigma"], seasonal["xi"])],
        "linear occurrence + stationary intensity":
            [(linear["pi"], stationary_sigma, pot_xi)],
        "linear occurrence + linear intensity":
            [(linear["pi"], linear["sigma"], linear["xi"])],
        "GP occurrence + stationary intensity":
            [(gp_pi[s], stationary_sigma, pot_xi) for s in range(len(gp_xi))],
        "GP occurrence + linear intensity":
            [(gp_pi[s], linear["sigma"], linear["xi"]) for s in range(len(gp_xi))],
        "linear occurrence + GP intensity":
            [(linear["pi"], gp_sigma[s], float(gp_xi[s])) for s in range(len(gp_xi))],
        "GP occurrence + GP intensity":
            [(gp_pi[s], gp_sigma[s], float(gp_xi[s])) for s in range(len(gp_xi))],
    }
    factorial_scores, factorial_rows, decomposition_rows = {}, [], []
    for name, specs in factorial_specs.items():
        pieces = [exact_twcrps_parts(index_test, *spec) for spec in specs]
        score = np.mean([p[0] for p in pieces], axis=0)
        obs = np.mean([p[1] for p in pieces], axis=0)
        reward = np.mean([p[2] for p in pieces], axis=0)
        penalty = np.mean([p[3] for p in pieces], axis=0)
        factorial_scores[name] = score
        factorial_rows.append(dict(city=city, model=name,
                                    mean_twCRPS=score.mean(),
                                    twCRPSS_vs_seasonal=1-score.mean()/seasonal_score.mean()))
        decomposition_rows.append(dict(city=city, model=name,
                                       observation_term=obs.mean(),
                                       event_reward=reward.mean(),
                                       non_event_and_tail_penalty=penalty.mean(),
                                       total=score.mean()))
    pd.DataFrame(factorial_rows).to_csv(OUT/f"{city}_factorial_scores.csv", index=False)
    pd.DataFrame(decomposition_rows).to_csv(OUT/f"{city}_twcrps_decomposition.csv", index=False)

    # Key controlled increments and paired intervals.
    contrasts = [
        ("linear occurrence added to seasonal intensity",
         "linear occurrence + seasonal intensity",
         "seasonal occurrence + seasonal intensity"),
        ("linear intensity added with occurrence fixed",
         "linear occurrence + linear intensity",
         "linear occurrence + seasonal intensity"),
        ("GP occurrence replacing linear occurrence; intensity fixed linear",
         "GP occurrence + linear intensity",
         "linear occurrence + linear intensity"),
        ("GP intensity replacing linear intensity; occurrence fixed linear",
         "linear occurrence + GP intensity",
         "linear occurrence + linear intensity"),
        ("GP intensity added with GP occurrence fixed",
         "GP occurrence + GP intensity",
         "GP occurrence + stationary intensity"),
    ]
    contrast_rows = []
    for label, new_name, old_name in contrasts:
        new, old = factorial_scores[new_name], factorial_scores[old_name]
        lo, hi = paired_interval(new, old, seasonal_score, block=14)
        contrast_rows.append(dict(city=city, contrast=label,
                                  gain=(old.mean()-new.mean())/seasonal_score.mean(),
                                  ci_low=lo, ci_high=hi))
    pd.DataFrame(contrast_rows).to_csv(OUT/f"{city}_factorial_contrasts.csv", index=False)

    # Higher verification thresholds.
    threshold_rows = []
    for tau in (1.0, 1.25, 1.5, 2.0):
        sref, *_ = exact_twcrps_parts(index_test, seasonal["pi"],
                                      seasonal["sigma"], seasonal["xi"], tau=tau)
        slin, *_ = exact_twcrps_parts(index_test, linear["pi"],
                                      linear["sigma"], linear["xi"], tau=tau)
        sgp = np.mean([exact_twcrps_parts(index_test, gp_pi[s], gp_sigma[s],
                                          float(gp_xi[s]), tau=tau)[0]
                       for s in range(len(gp_xi))], axis=0)
        threshold_rows.append(dict(city=city, tau=tau,
                                   n_test_above_tau=int((index_test > tau).sum()),
                                   mean_reference=sref.mean(),
                                   mean_linear=slin.mean(), mean_gp=sgp.mean(),
                                   linear_skill=1-slin.mean()/sref.mean(),
                                   gp_skill=1-sgp.mean()/sref.mean(),
                                   gp_minus_linear=(slin.mean()-sgp.mean())/sref.mean()))
    pd.DataFrame(threshold_rows).to_csv(OUT/f"{city}_verification_thresholds.csv", index=False)

    # Block-length sensitivity for the most important full-sample differences.
    bootstrap_rows = []
    for block in (7, 14, 30, 90):
        for label, new, old in (
            ("GP minus linear", gp_score, linear_score),
            ("GP minus seasonal", gp_score, seasonal_score),
            ("linear minus seasonal", linear_score, seasonal_score)):
            lo, hi = paired_interval(new, old, seasonal_score, block=block,
                                     seed=20261022+block)
            bootstrap_rows.append(dict(city=city, comparison=label,
                                       block_days=block,
                                       estimate=(old.mean()-new.mean())/seasonal_score.mean(),
                                       ci_low=lo, ci_high=hi))

    # Conditional score: event-order blocks versus calendar-time blocks.
    ext = event_test > 0.5
    yext = index_test[ext]-U
    cond_linear, *_ = exact_twcrps_parts(yext, np.ones(ext.sum()),
                                         linear["sigma"][ext], linear["xi"],
                                         tau=0.0, u=0.0)
    cond_gp = np.mean([exact_twcrps_parts(
        yext, np.ones(ext.sum()), gp_sigma[s][ext], float(gp_xi[s]),
        tau=0.0, u=0.0)[0] for s in range(len(gp_xi))], axis=0)
    # Event-order block, expressed as raw CRPS difference.
    rng = np.random.default_rng(20261022)
    vals = []
    for _ in range(N_BOOT):
        ii = moving_block_indices(ext.sum(), rng, 10)
        vals.append(np.mean(cond_linear[ii]-cond_gp[ii]))
    lo, hi = np.quantile(vals, [.025, .975])
    bootstrap_rows.append(dict(city=city, comparison="conditional GP minus linear",
                               block_days="10 extreme observations",
                               estimate=np.mean(cond_linear-cond_gp),
                               ci_low=lo, ci_high=hi))
    for block in (14, 30, 90):
        lo, hi = conditional_calendar_interval(cond_gp, cond_linear, ext, block,
                                               seed=20261022+block)
        bootstrap_rows.append(dict(city=city, comparison="conditional GP minus linear",
                                   block_days=f"{block} calendar days",
                                   estimate=np.mean(cond_linear-cond_gp),
                                   ci_low=lo, ci_high=hi))
    pd.DataFrame(bootstrap_rows).to_csv(OUT/f"{city}_bootstrap_sensitivity.csv", index=False)

    # Absolute scores and calibration diagnostics.
    abs_rows = []
    occurrence_models = {
        "seasonal": seasonal["pi"], "penalised linear": linear["pi"],
        "GP seed-average probability": gp_pi_mean,
    }
    tail_models = {"seasonal": seasonal_score, "penalised linear": linear_score,
                   "GP average seed score": gp_score}
    bs_ref = np.mean((seasonal["pi"]-event_test)**2)
    for name, p in occurrence_models.items():
        terms = reliability_terms(event_test, p)
        abs_rows.append(dict(city=city, model=name, mean_BS=terms["BS"],
                             BSS_vs_seasonal=1-terms["BS"]/bs_ref,
                             reliability=terms["REL"], resolution=terms["RES"],
                             uncertainty=terms["UNC"], mean_log_score=terms["LogS"],
                             mean_twCRPS=tail_models[name if name != "GP seed-average probability"
                                                     else "GP average seed score"].mean(),
                             twCRPSS_vs_seasonal=1-tail_models[name if name != "GP seed-average probability"
                                                               else "GP average seed score"].mean()
                             /seasonal_score.mean()))
    pd.DataFrame(abs_rows).to_csv(OUT/f"{city}_absolute_scores_calibration.csv", index=False)

    # Numerical-grid validation against the exact deterministic formula.
    zmax = float(max(data["I"].numpy())*1.05)
    numeric = ev.twcrps_model(index_test, linear["pi"], linear["sigma"],
                              linear["xi"], u=U, tau=U, zmax=zmax, nz=1200)
    validation = dict(city=city, exact_mean=float(linear_score.mean()),
                      numerical_mean=float(numeric.mean()),
                      difference=float(numeric.mean()-linear_score.mean()),
                      zmax=zmax, nz=1200)
    (OUT/f"{city}_exact_score_validation.json").write_text(json.dumps(validation, indent=2))

    summary = dict(city=city, n_train=cut, n_test=len(index_test),
                   n_test_extremes=int(ext.sum()),
                   selected_linear_lambda_occ=linear["lambda_occ"],
                   selected_linear_lambda_tail=linear["lambda_tail"],
                   linear_xi=linear["xi"], gp_xi=gp_xi.tolist(),
                   seasonal_optimizer_success=seasonal["optimiser_ok"])
    (OUT/f"{city}_analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(contrast_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(threshold_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    settings = dict(occurrence_lambdas=OCC_LAMBDAS,
                    tail_lambdas=TAIL_LAMBDAS, n_boot=N_BOOT,
                    bootstrap_interval="percentile 95%",
                    main_split="1989-2008 train; 2009-2018 test")
    (OUT/"analysis_settings.json").write_text(json.dumps(settings, indent=2))
    for city_name in ("london", "paris"):
        analyse_city(city_name)
