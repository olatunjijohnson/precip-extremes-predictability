"""Core forecast-attribution analyses.

This script uses the fixed 1989--2008 training and 2009--2018 test split.  It
produces machine-readable CSV tables in ``results``.

Timing safeguards
-----------------
* Monthly NAO/AO/Nino3.4 values are lagged by one calendar month.
* Cached daily ERA5 aggregates are lagged by one additional day because their
  hourly composition cannot be cut at 21 UTC after aggregation.
* The city index at t-1 is the latest response available for a forecast of t.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import genpareto
from sklearn.linear_model import LogisticRegression

import evaluation as ev
from data import load_drivers, load_real, standardize
from distreg import UnifiedHurdle
from tcdgp import TCDGP


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DATA = HERE.parent / "data"
U = 1.0
N_BOOT = 2000
BLOCK = 14
INTERVAL_LEVEL = 0.95
SEEDS = (0, 1, 2, 3, 4)


def moving_block_indices(n, rng, block=BLOCK):
    starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def paired_gain_ci(score_new, score_old, denominator, seed=20261015,
                   block=BLOCK):
    """Gain in skill of new over old, using a common score denominator."""
    new, old, den = map(lambda x: np.asarray(x, float),
                        (score_new, score_old, denominator))
    point = (old.mean() - new.mean()) / den.mean()
    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = moving_block_indices(len(new), rng, block=block)
        vals[b] = (old[ii].mean() - new[ii].mean()) / den[ii].mean()
    alpha = (1.0 - INTERVAL_LEVEL) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return point, lo, hi


def skill_ci(score_model, score_ref, seed=20261015, block=BLOCK):
    score_model, score_ref = map(lambda x: np.asarray(x, float),
                                 (score_model, score_ref))
    point = 1 - score_model.mean() / score_ref.mean()
    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = moving_block_indices(len(score_model), rng, block=block)
        vals[b] = 1 - score_model[ii].mean() / score_ref[ii].mean()
    alpha = (1.0 - INTERVAL_LEVEL) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return point, lo, hi


def city_paths(city):
    return {
        "index": DATA / f"{city}_precip_extreme_index_95th_1989_2018.csv",
        "era": DATA / f"era5_{city}_daily.parquet",
        "conv": DATA / ("era5_cape_ivt_daily.parquet" if city == "london"
                         else "era5_paris_cape_ivt_daily.parquet"),
        "tele": DATA / "teleconnections_daily.csv",
    }


def stage_drivers(city):
    p = city_paths(city)
    tele = load_drivers(p["tele"], lag_monthly_teleconnections=True)
    era = load_drivers(p["era"], extra_era5_lag_days=1)
    conv = load_drivers(p["conv"], extra_era5_lag_days=1)
    return [
        ("index history", None),
        ("+ previous-month teleconnections", tele),
        ("+ circulation and moisture", pd.concat([tele, era], axis=1)),
        ("+ convective instability", pd.concat([tele, era, conv], axis=1)),
    ]


def load_stage(city, drivers):
    d, dates = load_real(city_paths(city)["index"], n_lags=14, u=U,
                         horizon=1, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    xtr, xte = standardize(d["X"][:cut], d["X"][cut:])
    return d, dates, cut, xtr, xte


def seasonal_event_prob(o_train, train_dates, test_dates):
    tab = pd.DataFrame({"o": o_train}, index=train_dates).groupby(
        lambda x: x.month)["o"].agg(["sum", "count"])
    # Jeffreys smoothing avoids exact zero/one monthly probabilities.
    rate = (tab["sum"] + 0.5) / (tab["count"] + 1.0)
    return np.array([rate.loc[d.month] for d in test_dates], float)


def fit_seasonal_gpd(index_train, train_dates):
    mask = index_train > U
    y = index_train[mask] - U
    doy = train_dates[mask].dayofyear.to_numpy()
    z = np.column_stack([np.ones(len(y)), np.sin(2*np.pi*doy/365.25),
                         np.cos(2*np.pi*doy/365.25)])

    def nll(theta):
        beta, xi = theta[:3], 0.5*np.tanh(theta[3])
        sigma = np.exp(np.clip(z @ beta, -8, 8))
        base = 1 + xi*y/sigma
        if np.any(base <= 0):
            return 1e12
        if abs(xi) < 1e-5:
            ll = -np.log(sigma) - y/sigma
        else:
            ll = -np.log(sigma) - (1/xi + 1)*np.log(base)
        return -ll.sum()

    xi0, _, sig0 = genpareto.fit(y, floc=0)
    start = np.array([np.log(sig0), 0, 0, np.arctanh(np.clip(xi0/0.5, -.95, .95))])
    # Nelder--Mead is stable here; gradient-based fits can stop with a
    # precision-loss warning near xi=0 even when the likelihood is well behaved.
    fit = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    return fit.x, bool(fit.success)


def seasonal_pot_scores(i_test, test_dates, p_seasonal, theta, zmax):
    beta, xi = theta[:3], 0.5*np.tanh(theta[3])
    doy = test_dates.dayofyear.to_numpy()
    z = np.column_stack([np.ones(len(doy)), np.sin(2*np.pi*doy/365.25),
                         np.cos(2*np.pi*doy/365.25)])
    sigma = np.exp(np.clip(z @ beta, -8, 8))
    return ev.twcrps_model(i_test, p_seasonal, sigma, xi, u=U, tau=U,
                           zmax=zmax, nz=1200), xi


def fit_linear_hurdle(xtr, d, cut, xte):
    torch.manual_seed(0)
    model = UnifiedHurdle(xtr.shape[1], u=U)
    model.fit(xtr, d["O"][:cut], d["excess"][:cut], epochs=1000,
              lr=0.02, weight_decay=1e-3, verbose=False)
    return model.predict(xte), model


def analyse_city(city, run_gp=True):
    print(f"\n===== {city.upper()} =====", flush=True)
    stages = stage_drivers(city)
    stage_rows, predictions = [], []
    base_d, dates, cut, _, _ = load_stage(city, None)
    train_dates, test_dates = dates[:cut], dates[cut:]
    i_tr, i_te = base_d["I"][:cut].numpy(), base_d["I"][cut:].numpy()
    o_tr, o_te = base_d["O"][:cut].numpy(), base_d["O"][cut:].numpy()
    zmax = float(max(i_tr.max(), i_te.max()) * 1.05)

    p_const = np.full(len(o_te), o_tr.mean())
    bs_const = (p_const - o_te)**2
    p_seas = seasonal_event_prob(o_tr, train_dates, test_dates)
    bs_seas = (p_seas - o_te)**2
    pot = ev.fit_pot(i_tr, U)
    tw_stat = ev.twcrps_pot(i_te, *pot, u=U, tau=U, zmax=zmax, nz=1200)
    theta, opt_ok = fit_seasonal_gpd(i_tr, train_dates)
    tw_seas, xi_seas = seasonal_pot_scores(i_te, test_dates, p_seas, theta, zmax)

    ref_rows = []
    for name, scores, ref in [
        ("seasonal event rate vs constant rate", bs_seas, bs_const),
        ("seasonal POT vs stationary POT", tw_seas, tw_stat),
    ]:
        pt, lo, hi = skill_ci(scores, ref)
        ref_rows.append(dict(city=city, comparison=name, skill=pt,
                             ci_low=lo, ci_high=hi))
    print(pd.DataFrame(ref_rows).to_string(index=False), flush=True)

    for stage, drv in stages:
        d, _, c, xtr, xte = load_stage(city, drv)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(
            xtr.numpy(), d["O"][:c].numpy())
        p = lr.predict_proba(xte.numpy())[:, 1]
        bs = (p - o_te)**2
        (pi, sigma, xi), hurdle = fit_linear_hurdle(xtr, d, c, xte)
        tw = ev.twcrps_model(i_te, pi, sigma, xi, u=U, tau=U,
                             zmax=zmax, nz=1200)
        bss = 1 - bs.mean()/bs_seas.mean()
        twss = 1 - tw.mean()/tw_seas.mean()
        stage_rows.append(dict(city=city, stage=stage, n_predictors=xtr.shape[1],
                               BSS_vs_seasonal=bss, twCRPSS_vs_seasonal=twss,
                               linear_hurdle_xi=xi))
        predictions.append(dict(stage=stage, bs=bs, tw=tw, p=p, pi=pi,
                                sigma=sigma, xi=xi, model=hurdle))
        print(stage_rows[-1], flush=True)

    gain_rows = []
    for k in range(1, len(predictions)):
        prev, new = predictions[k-1], predictions[k]
        for metric, den, label in [("BSS", bs_seas, "bs"),
                                   ("twCRPSS", tw_seas, "tw")]:
            pt, lo, hi = paired_gain_ci(new[label], prev[label], den,
                                        seed=20261015+k)
            gain_rows.append(dict(city=city, added_stage=stages[k][0],
                                  metric=metric, gain=pt,
                                  ci_low=lo, ci_high=hi))
    print(pd.DataFrame(gain_rows).to_string(index=False), flush=True)

    # Like-for-like tail comparison. Occurrence-only logistic is completed with
    # the stationary GPD; all models are evaluated on identical days and grid.
    full = predictions[-1]
    tw_logstat = ev.twcrps_model(i_te, full["p"], np.full(len(i_te), pot[1]),
                                 pot[2], u=U, tau=U, zmax=zmax, nz=1200)
    competitors = {
        "seasonal POT": tw_seas,
        "index-history linear hurdle": predictions[0]["tw"],
        "logistic + stationary GPD": tw_logstat,
        "regularised linear hurdle": full["tw"],
    }

    gp_meta = []
    if run_gp:
        d, _, c, xtr, xte = load_stage(city, stages[-1][1])
        gp_scores, gp_bs, gp_xi, gp_cond_scores = [], [], [], []
        ext = o_te > 0.5
        y_ext = i_te[ext] - U
        zmax_ext = float(y_ext.max() * 1.05)
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = TCDGP(d_in=xtr.shape[1], u=U, M=64, Q=2,
                          coupled=True, deep=False)
            model.fit(xtr, d["O"][:c], d["excess"][:c], epochs=600,
                      lr=0.02, batch=1024, verbose=False)
            pars = model.predict_params(xte)
            pi = pars["pi"].numpy(); sigma = pars["sigma"].numpy()
            xi = float(pars["xi"])
            gp_scores.append(ev.twcrps_model(i_te, pi, sigma, xi, u=U,
                                             tau=U, zmax=zmax, nz=1200))
            gp_cond_scores.append(ev.twcrps_model(
                y_ext, np.ones(ext.sum()), sigma[ext], xi,
                u=0.0, tau=0.0, zmax=zmax_ext, nz=1200))
            gp_bs.append((pi-o_te)**2); gp_xi.append(xi)
            print(f"GP seed {seed}: xi={xi:.4f}", flush=True)
        gp_tw = np.mean(gp_scores, axis=0)
        gp_brier = np.mean(gp_bs, axis=0)
        gp_meta = gp_xi
        model_rows = []
        for name, score in competitors.items():
            sk, lo, hi = skill_ci(score, tw_seas)
            diff, dlo, dhi = paired_gain_ci(gp_tw, score, tw_seas)
            model_rows.append(dict(city=city, model=name,
                                   twCRPSS_vs_seasonal=sk,
                                   GP_minus_model=diff,
                                   diff_ci_low=dlo, diff_ci_high=dhi))
        sk, lo, hi = skill_ci(gp_tw, tw_seas)
        model_rows.append(dict(city=city, model="GP-EVT hurdle",
                               twCRPSS_vs_seasonal=sk, GP_minus_model=0.0,
                               diff_ci_low=0.0, diff_ci_high=0.0))
        print(pd.DataFrame(model_rows).to_string(index=False), flush=True)

        # Conditional excess comparison on the same observed exceedance days.
        # Setting pi=1 and u=0 scores only the fitted GPD excess distribution.
        cond_stat = ev.twcrps_model(
            y_ext, np.ones(ext.sum()), np.full(ext.sum(), pot[1]), pot[2],
            u=0.0, tau=0.0, zmax=zmax_ext, nz=1200)
        cond_linear = ev.twcrps_model(
            y_ext, np.ones(ext.sum()), full["sigma"][ext], full["xi"],
            u=0.0, tau=0.0, zmax=zmax_ext, nz=1200)
        cond_gp = np.mean(gp_cond_scores, axis=0)
        conditional_rows = [dict(city=city, model="stationary GPD",
                                 CRPSS_vs_stationary=0.0,
                                 GP_minus_model=np.nan,
                                 diff_ci_low=np.nan, diff_ci_high=np.nan)]
        for name, score in (("regularised linear hurdle", cond_linear),
                            ("GP-EVT hurdle", cond_gp)):
            sk, lo, hi = skill_ci(score, cond_stat, block=10)
            if name == "regularised linear hurdle":
                diff, dlo, dhi = paired_gain_ci(cond_gp, score, cond_stat,
                                                block=10)
            else:
                diff, dlo, dhi = 0.0, 0.0, 0.0
            conditional_rows.append(dict(
                city=city, model=name, CRPSS_vs_stationary=sk,
                skill_ci_low=lo, skill_ci_high=hi,
                GP_minus_model=diff, diff_ci_low=dlo, diff_ci_high=dhi))
        print(pd.DataFrame(conditional_rows).to_string(index=False), flush=True)
    else:
        model_rows = []
        conditional_rows = []

    pd.DataFrame(ref_rows).to_csv(OUT/f"{city}_stronger_references.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(OUT/f"{city}_predictor_stages.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(OUT/f"{city}_stage_gains.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT/f"{city}_model_comparison.csv", index=False)
    pd.DataFrame(conditional_rows).to_csv(
        OUT/f"{city}_conditional_excess_models.csv", index=False)
    meta = dict(city=city, n_train=cut, n_test=len(i_te),
                train_extremes=int(o_tr.sum()), test_extremes=int(o_te.sum()),
                stationary_xi=pot[2], seasonal_xi=xi_seas,
                seasonal_optimizer_success=opt_ok, gp_xi_by_seed=gp_meta,
                n_boot=N_BOOT, interval_level=INTERVAL_LEVEL,
                bootstrap_block_days=BLOCK,
                gp_seeds=list(SEEDS))
    (OUT/f"{city}_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    for CITY in ("london", "paris"):
        analyse_city(CITY, run_gp=True)
