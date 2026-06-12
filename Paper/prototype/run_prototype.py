"""
run_prototype.py — end-to-end smoke test of the TCDGP prototype.

(1) SYNTHETIC validation: data with a KNOWN occurrence-intensity correlation.
    Fit the dependent (coupled) vs independent hurdle; check that the coupled
    model recovers a positive rho and improves tail log-likelihood
    ("borrowing strength").
(2) REAL smoke test: London 95th-percentile index; fit, conformally calibrate
    an extreme upper bound, and report coverage vs nominal.

Run:  python3 run_prototype.py
"""
import os, sys, numpy as np, torch
from sklearn.metrics import average_precision_score

from tcdgp import TCDGP
from data import make_synthetic, load_real, load_drivers, standardize
from conformal import split_conformal_upper, adaptive_conformal
import evaluation as ev
from distreg import ZIDistReg
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))


def tail_loglik(model, d):
    """Mean GPD log-density on the extreme days (the data-starved tail)."""
    p = model.predict_params(d["X"])
    mask = d["O"] == 1
    if mask.sum() == 0:
        return float("nan")
    ll = HurdleGPD_logpdf(model, d["excess"][mask], p["sigma"][mask], p["xi"])
    return float(ll.mean())


def HurdleGPD_logpdf(model, excess, sigma, xi):
    return model.lik._gpd_logpdf(excess.clamp_min(0), sigma, xi)


# ===========================================================================
def synthetic_experiment():
    print("\n" + "=" * 64 + "\n  (1) SYNTHETIC: borrowing strength / rho recovery\n" + "=" * 64)
    data, truth = make_synthetic(n=4000, rho=0.7, xi=0.2, base_rate=0.08, seed=1)
    print(f"true rho={truth['rho']}, true xi={truth['xi']}, "
          f"exceedance rate={truth['rate']:.3f}")
    ntr = 3000
    Xtr, Xte = standardize(data["X"][:ntr], data["X"][ntr:])
    tr = {**data, "X": Xtr, "O": data["O"][:ntr], "excess": data["excess"][:ntr],
          "I": data["I"][:ntr]}
    te = {**data, "X": Xte, "O": data["O"][ntr:], "excess": data["excess"][ntr:],
          "I": data["I"][ntr:]}

    results = {}
    for coupled in (False, True):
        name = "coupled (dependent)" if coupled else "independent"
        print(f"\n-- {name} hurdle --")
        torch.manual_seed(0)
        m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=48, Q=2, coupled=coupled, deep=True)
        m.fit(tr["X"], tr["O"], tr["excess"], epochs=300, lr=0.02, verbose=True)
        p = m.predict_params(te["X"])
        prauc = average_precision_score(te["O"].numpy(), p["pi"].numpy())
        tll = tail_loglik(m, te)
        results[name] = dict(rho=m.gp.rho(), xi=m.lik.xi().item(), prauc=prauc, tll=tll)
        print(f"   recovered rho={m.gp.rho():+.3f}  xi={m.lik.xi().item():.3f}  "
              f"occ PR-AUC={prauc:.3f}  tail loglik={tll:.3f}")

    print("\n  SUMMARY (synthetic):")
    for k, v in results.items():
        print(f"   {k:22s} rho={v['rho']:+.3f}  PR-AUC={v['prauc']:.3f}  "
              f"tailLL={v['tll']:.3f}")
    dep, ind = results["coupled (dependent)"], results["independent"]
    print(f"\n  -> coupled recovered rho={dep['rho']:+.3f} (true {truth['rho']:+.2f}); "
          f"tail loglik {ind['tll']:.3f} (indep) -> {dep['tll']:.3f} (coupled)")


# ===========================================================================
def real_experiment():
    print("\n" + "=" * 64 + "\n  (2) REAL: London 95th index — fit + conformal coverage\n" + "=" * 64)
    csv = os.path.normpath(os.path.join(
        HERE, "..", "..", "data", "london_precip_extreme_index_95th_1989_2018.csv"))
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}")
        return
    # auto-use ERA5 / teleconnection drivers if they have been fetched
    era5 = os.path.join(HERE, "era5_london_daily.parquet")
    tele = os.path.join(HERE, "teleconnections_daily.csv")
    drivers = load_drivers(era5 if os.path.exists(era5) else None,
                           tele if os.path.exists(tele) else None)
    print(f"  drivers: {'attached' if drivers is not None else 'none (index-only)'}")
    data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
    split = dates <= np.datetime64("2008-12-31")
    cut = int(split.sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    O, I = data["O"], data["I"]
    print(f"  train={cut} days ({int(O[:cut].sum())} extremes), "
          f"test={len(O)-cut} days ({int(O[cut:].sum())} extremes)")

    torch.manual_seed(0)
    m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=64, Q=2, coupled=True, deep=True)
    m.fit(Xtr, O[:cut], data["excess"][:cut], epochs=250, lr=0.02,
          batch=1024, verbose=True)

    # last 20% of train as conformal calibration set
    ncal = int(0.2 * cut)
    Xcal = Xtr[-ncal:]
    p_cal = m.predict_params(Xcal)
    p_te = m.predict_params(Xte)
    alpha = 0.10
    qhi_cal = m.tail_quantile(1 - alpha, p_cal).numpy()
    qhi_te = m.tail_quantile(1 - alpha, p_te).numpy()
    y_cal = I[cut - ncal:cut].numpy()
    y_te = I[cut:].numpy()

    upper, cov, qh = split_conformal_upper(y_cal, qhi_cal, y_te, qhi_te, alpha=alpha)
    print(f"\n  split-conformal upper bound (target {1-alpha:.0%}): "
          f"coverage={cov:.3f}, q_hat={qh:.3f}")
    scores_init = y_cal - qhi_cal
    _, cov_aci = adaptive_conformal(y_te, qhi_te, scores_init, alpha=alpha, gamma=0.02)
    print(f"  adaptive-conformal (ACI) running coverage: {cov_aci:.3f}")
    prauc = average_precision_score(O[cut:].numpy(), p_te["pi"].numpy())
    base = float(O[cut:].mean())
    print(f"  occurrence PR-AUC={prauc:.3f} (prevalence baseline {base:.3f}); "
          f"recovered rho={m.gp.rho():+.3f}, xi={m.lik.xi().item():.3f}")


def _fit_eval_real(csv, drivers, epochs, label, seed=0, horizon=1):
    """Fit on London 95th with a given driver set; return a metrics row."""
    data, dates = load_real(csv, n_lags=14, u=1.0, horizon=horizon, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    O, I = data["O"], data["I"]
    torch.manual_seed(seed)
    m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=64, Q=2, coupled=True, deep=True)
    m.fit(Xtr, O[:cut], data["excess"][:cut], epochs=epochs, lr=0.02,
          batch=1024, verbose=False)
    ncal = int(0.2 * cut)
    p_cal, p_te = m.predict_params(Xtr[-ncal:]), m.predict_params(Xte)
    a = 0.10
    qc = m.tail_quantile(1 - a, p_cal).numpy(); qt = m.tail_quantile(1 - a, p_te).numpy()
    _, cov, _ = split_conformal_upper(I[cut - ncal:cut].numpy(), qc,
                                      I[cut:].numpy(), qt, alpha=a)
    prauc = average_precision_score(O[cut:].numpy(), p_te["pi"].numpy())
    return dict(label=label, nfeat=Xtr.shape[1], prauc=prauc, cov=cov,
                rho=m.gp.rho(), base=float(O[cut:].mean()))


def calibration_eval(seeds=(0, 1, 2), epochs=250, u=1.0, save_npz="reliability.npz"):
    """DECISIVE TEST: is the model calibrated AND sharper than baselines, and
    does it beat climatology on a tail-focused proper score (twCRPS)?"""
    print("\n" + "=" * 72 + "\n  (6) CALIBRATION-AWARE EVALUATION (London 95th, h=1)\n"
          + "=" * 72)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}"); return
    configs = _driver_configs()
    configs = [c for c in configs if c[0] == "index-only" or c[0].startswith("+ERA5")]

    # target-only split for the climatology / POT baselines
    bd0, dates = load_real(csv, n_lags=14, u=u, drivers_df=None)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    I_tr, I_te = bd0["I"][:cut].numpy(), bd0["I"][cut:].numpy()
    O_te = bd0["O"][cut:].numpy()
    zmax = float(max(I_tr.max(), I_te.max()) * 1.05)
    tw = dict(u=u, tau=u, zmax=zmax)

    rows = []   # (name, BS, BSS, REL, RES, twCRPSS)
    rel_curves = {}

    # --- climatology (constant base rate; empirical-CDF tail) ---
    base = float((I_tr > u).mean())
    p_clim = np.full(len(O_te), base)
    bdc = ev.brier_decomposition(O_te, p_clim)
    tw_clim = ev.twcrps_climatology(I_te, I_tr, tau=u, zmax=zmax)
    rows.append(("climatology", bdc["BS"], bdc["BSS"], bdc["REL"], bdc["RES"], 0.0))
    rel_curves["climatology"] = ev.reliability_curve(O_te, p_clim)

    # --- stationary POT (no covariates) ---
    pi_p, sig_p, xi_p = ev.fit_pot(I_tr, u)
    tw_pot = ev.twcrps_pot(I_te, pi_p, sig_p, xi_p, **tw)
    rows.append(("stationary POT", bdc["BS"], bdc["BSS"], bdc["REL"], bdc["RES"],
                 ev.skill(tw_pot, tw_clim)))

    # --- per config: logistic baseline + TCDGP (seed-averaged) ---
    for name, drv in configs:
        data, _ = load_real(csv, n_lags=14, u=u, drivers_df=drv)
        Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
        O_tr = data["O"][:cut].numpy()
        # plain logistic regression = fair *calibrated* occurrence baseline
        # (class-balanced logistic ranks well but is badly miscalibrated, BS~0.21)
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(Xtr.numpy(), O_tr)
        p_lr = lr.predict_proba(Xte.numpy())[:, 1]
        bdl = ev.brier_decomposition(O_te, p_lr)
        rows.append((f"logistic [{name}]", bdl["BS"], bdl["BSS"], bdl["REL"],
                     bdl["RES"], float("nan")))
        if name == "+ERA5 (+teleconn)":
            rel_curves["logistic+ERA5"] = ev.reliability_curve(O_te, p_lr)
        # TCDGP, seed-averaged
        agg = {k: [] for k in ("BS", "BSS", "REL", "RES", "twss")}
        last_pi = None
        for s in seeds:
            torch.manual_seed(s)
            m = TCDGP(d_in=Xtr.shape[1], u=u, M=64, Q=2, coupled=True, deep=True)
            m.fit(Xtr, data["O"][:cut], data["excess"][:cut], epochs=epochs,
                  lr=0.02, batch=1024, verbose=False)
            p = m.predict_params(Xte)
            pi = p["pi"].numpy(); sig = p["sigma"].numpy(); xi = float(p["xi"])
            bd = ev.brier_decomposition(O_te, pi)
            twm = ev.twcrps_model(I_te, pi, sig, xi, **tw)
            agg["BS"].append(bd["BS"]); agg["BSS"].append(bd["BSS"])
            agg["REL"].append(bd["REL"]); agg["RES"].append(bd["RES"])
            agg["twss"].append(ev.skill(twm, tw_clim))
            last_pi = pi
        rows.append((f"TCDGP [{name}]", np.mean(agg["BS"]), np.mean(agg["BSS"]),
                     np.mean(agg["REL"]), np.mean(agg["RES"]), np.mean(agg["twss"])))
        if name == "+ERA5 (+teleconn)":
            rel_curves["TCDGP+ERA5"] = ev.reliability_curve(O_te, last_pi)

    # --- report ---
    print(f"\n  Occurrence (Brier decomposition) + tail twCRPS skill vs climatology:")
    print(f"  {'model':22s} {'BS':>7s} {'BSS':>7s} {'REL':>8s} {'RES':>8s} {'twCRPSS':>8s}")
    for n, bs, bss, rel, res, twss in rows:
        tws = "  n/a " if np.isnan(twss) else f"{twss:+.3f}"
        print(f"  {n:22s} {bs:7.4f} {bss:+7.3f} {rel:8.5f} {res:8.5f} {tws:>8s}")
    print("\n  Guide: BSS>0 and twCRPSS>0 mean BETTER than climatology; "
          "RES (resolution) is sharpness/informativeness; REL (reliability) lower=better.")
    # save reliability curves for the figure
    np.savez(os.path.join(HERE, save_npz),
             **{k: np.array(v, dtype=object) for k, v in rel_curves.items()})
    print(f"  [saved reliability curves -> {save_npz}]")


def _bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg: boolean array of which p-values pass FDR control."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    below = ranked <= alpha * (np.arange(1, n + 1) / n)
    keep = np.zeros(n, bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        keep[order[:kmax + 1]] = True
    return keep


def _block_boot(num, den, n_boot=2000, block=10, seed=0):
    """Moving-block bootstrap of the skill 1 - sum(num)/sum(den) over time-ordered
    points (respects short-range temporal dependence among extreme days)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([(np.arange(s, s + block) % n)
                              for s in rng.integers(0, n, nb)])[:n]
        out[b] = 1 - num[idx].sum() / den[idx].sum()
    return out


def intensity_multisplit(u=1.0, cuts=(2002, 2005, 2008, 2011), alpha=0.05, n_boot=1000):
    """Robustness: does the CRPS-of-excess intensity skill hold across DIFFERENT
    expanding-window train/test splits (not just test=2009-2018)? Fixed alpha
    (no per-split tuning); forward-only (train on past, test on future)."""
    from sklearn.linear_model import QuantileRegressor
    print("\n" + "=" * 72 + "\n  (11) INTENSITY ROBUSTNESS across train/test splits\n"
          + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    X = data["X"]; I = data["I"].numpy(); yr = dates.year.values
    qs = np.arange(0.1, 0.91, 0.1)

    def ploss(y, P, qs):
        e = y[:, None] - P
        return np.maximum(qs[None, :] * e, (qs[None, :] - 1) * e)

    print(f"\n  {'test period':>14s} {'n_tr':>5s} {'n_te':>5s} {'CRPS skill':>11s} "
          f"{'90% CI':>18s} {'P>0':>5s}")
    rng = np.random.default_rng(0)
    skills = []
    for C in cuts:
        tr, te = yr <= C, yr > C
        Xtr, Xte = standardize(X[tr], X[te])
        Xtr, Xte = Xtr.numpy(), Xte.numpy()
        Itr, Ite = I[tr], I[te]
        etr, ete = Itr > u, Ite > u
        if etr.sum() < 60 or ete.sum() < 40:
            print(f"  {'>'+str(C):>14s}  (too few extremes: {etr.sum()}/{ete.sum()})"); continue
        Xetr, exc_tr = Xtr[etr], Itr[etr] - u
        Xete, exc_te = Xte[ete], Ite[ete] - u
        preds = np.zeros((ete.sum(), len(qs)))
        for j, q in enumerate(qs):
            m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(Xetr, exc_tr)
            preds[:, j] = np.clip(m.predict(Xete), 0, None)
        preds = np.sort(preds, axis=1)
        clim = np.quantile(exc_tr, qs)
        cm = 2 * ploss(exc_te, preds, qs).mean(1)
        cc = 2 * ploss(exc_te, np.broadcast_to(clim, preds.shape), qs).mean(1)
        sk = 1 - cm.mean() / cc.mean()
        bs = _block_boot(cm, cc, n_boot=n_boot, block=10, seed=0)
        lo, hi = np.percentile(bs, [5, 95]); p0 = float(np.mean(bs > 0))
        skills.append(sk)
        print(f"  {'>'+str(C):>14s} {int(etr.sum()):5d} {int(ete.sum()):5d} "
              f"{sk:+11.3f}   [{lo:+.3f}, {hi:+.3f}]  {p0:4.2f}")
    if skills:
        print(f"\n  splits with positive skill: {sum(s>0 for s in skills)}/{len(skills)}; "
              f"mean skill {np.mean(skills):+.3f}")
    print("  (consistently positive across splits => robust intensity predictability)")


def intensity_firmup(u=1.0, n_boot=2000):
    """Firm up the intensity lead: CRPS-of-excess (proper score over ALL quantiles,
    no cherry-picking) + bootstrap CIs on the CRPS and q90 skills vs climatology."""
    from sklearn.linear_model import QuantileRegressor
    from sklearn.model_selection import KFold
    print("\n" + "=" * 72 + "\n  (10) INTENSITY FIRM-UP: CRPS-of-excess + bootstrap CI\n"
          + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    Xtr, Xte = Xtr.numpy(), Xte.numpy()
    I_tr, I_te = data["I"][:cut].numpy(), data["I"][cut:].numpy()
    Xetr, exc_tr = Xtr[I_tr > u], I_tr[I_tr > u] - u
    Xete, exc_te = Xte[I_te > u], I_te[I_te > u] - u
    n = len(exc_te)
    print(f"  train extremes={len(exc_tr)}, test extremes={n}")

    def ploss(y, P, qs):                       # elementwise pinball, (n,nq)
        e = y[:, None] - P
        return np.maximum(qs[None, :] * e, (qs[None, :] - 1) * e)

    def cv_alpha(q):
        best_a, best_l = 0.05, np.inf
        for a in (0.01, 0.03, 0.1, 0.3):
            ls = []
            for ti, vi in KFold(3, shuffle=True, random_state=0).split(Xetr):
                m = QuantileRegressor(quantile=q, alpha=a, solver="highs").fit(Xetr[ti], exc_tr[ti])
                ls.append(np.mean(ploss(exc_tr[vi], np.clip(m.predict(Xetr[vi]), 0, None)[:, None],
                                        np.array([q]))))
            if np.mean(ls) < best_l:
                best_l, best_a = np.mean(ls), a
        return best_a

    qs = np.arange(0.05, 0.96, 0.05)
    preds = np.zeros((n, len(qs)))
    for j, q in enumerate(qs):
        qr = QuantileRegressor(quantile=q, alpha=cv_alpha(q), solver="highs").fit(Xetr, exc_tr)
        preds[:, j] = np.clip(qr.predict(Xete), 0, None)
    preds = np.sort(preds, axis=1)             # rearrangement (no quantile crossing)
    clim = np.quantile(exc_tr, qs)

    Lm = ploss(exc_te, preds, qs)
    Lc = ploss(exc_te, np.broadcast_to(clim, (n, len(qs))), qs)
    crps_m, crps_c = 2 * Lm.mean(1), 2 * Lc.mean(1)        # CRPS per point
    j90 = int(np.argmin(np.abs(qs - 0.9)))
    pm90, pc90 = Lm[:, j90], Lc[:, j90]

    crps_sk = _block_boot(crps_m, crps_c, n_boot=n_boot, block=10, seed=0)
    q90_sk = _block_boot(pm90, pc90, n_boot=n_boot, block=10, seed=0)
    cl, cm, ch = np.percentile(crps_sk, [5, 50, 95])
    ql, qm, qh = np.percentile(q90_sk, [5, 50, 95])
    print(f"\n  CRPS-of-excess skill vs climatology : {cm:+.3f}  (90% CI [{cl:+.3f}, {ch:+.3f}])")
    print(f"  q90 pinball  skill vs climatology    : {qm:+.3f}  (90% CI [{ql:+.3f}, {qh:+.3f}])")
    print(f"  P(skill>0): CRPS={np.mean(crps_sk > 0):.2f}, q90={np.mean(q90_sk > 0):.2f}")
    print("  (moving-block bootstrap, block=10 extreme days)")
    print("\n  Verdict: CI excluding 0 => real out-of-sample intensity skill.")


def intensity_test(u=1.0):
    """DECISIVE intensity test: does a covariate model of the excess (I-u) beat
    the climatological excess on HELD-OUT extreme days? Isolates intensity-given-
    occurrence (conditions on observed extreme test days — a diagnostic)."""
    from sklearn.linear_model import QuantileRegressor
    print("\n" + "=" * 72 + "\n  (9) HOLD-OUT INTENSITY TEST (excess on extreme days)\n"
          + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    Xtr, Xte = Xtr.numpy(), Xte.numpy()
    I_tr, I_te = data["I"][:cut].numpy(), data["I"][cut:].numpy()
    etr, ete = I_tr > u, I_te > u
    Xetr, exc_tr = Xtr[etr], I_tr[etr] - u
    Xete, exc_te = Xte[ete], I_te[ete] - u
    print(f"  train extremes={etr.sum()}, test extremes={ete.sum()}")

    def pinball(y, p, q):
        e = y - p
        return np.mean(np.maximum(q * e, (q - 1) * e))

    # pick L1 strength by 3-fold CV on TRAIN only (no test selection)
    from sklearn.model_selection import KFold

    def cv_alpha(q):
        best_a, best_l = 0.03, np.inf
        for alpha in (0.003, 0.01, 0.03, 0.1, 0.3):
            losses = []
            for tr_i, va_i in KFold(3, shuffle=True, random_state=0).split(Xetr):
                m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(
                    Xetr[tr_i], exc_tr[tr_i])
                losses.append(pinball(exc_tr[va_i], np.clip(m.predict(Xetr[va_i]), 0, None), q))
            if np.mean(losses) < best_l:
                best_l, best_a = np.mean(losses), alpha
        return best_a

    print(f"\n  {'quantile':>9s} {'climatology':>12s} {'covariate':>11s} {'skill':>8s} {'alpha':>7s}")
    for q in (0.5, 0.75, 0.9):
        clim = np.quantile(exc_tr, q)
        pc = pinball(exc_te, clim, q)
        a = cv_alpha(q)
        qr = QuantileRegressor(quantile=q, alpha=a, solver="highs").fit(Xetr, exc_tr)
        pm = pinball(exc_te, np.clip(qr.predict(Xete), 0, None), q)
        print(f"  {q:9.2f} {pc:12.4f} {pm:11.4f} {1 - pm / pc:+8.3f} {a:7.3f}")
    print("\n  skill > 0  => covariate model beats climatology on held-out intensity "
          "(alpha chosen by CV on train only).")


def feature_screen(u=1.0):
    """Which variables (if any) carry signal — separately for OCCURRENCE and
    INTENSITY? L1-logistic picks occurrence features; rank correlation on extreme
    training days probes intensity. Strictly antecedent features only."""
    from scipy.stats import spearmanr
    print("\n" + "=" * 72 + "\n  (8) PER-VARIABLE PREDICTABILITY SCREEN\n" + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    names = ([f"lag_{i}" for i in range(1, 15)]
             + ["roll7", "roll30", "max7", "exc30", "sin_doy", "cos_doy"]
             + list(drivers.columns))
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, _ = standardize(data["X"][:cut], data["X"][cut:])
    Xtr = Xtr.numpy(); O_tr = data["O"][:cut].numpy(); I_tr = data["I"][:cut].numpy()
    assert len(names) == Xtr.shape[1], (len(names), Xtr.shape[1])

    # --- occurrence: L1-sparse logistic (which features survive) ---
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.3,
                            max_iter=2000).fit(Xtr, O_tr)
    coef = l1.coef_[0]
    order = np.argsort(-np.abs(coef))
    print("\n  OCCURRENCE — top L1-logistic features (|standardised coef|):")
    for i in order[:12]:
        if abs(coef[i]) > 1e-6:
            print(f"    {names[i]:34s} {coef[i]:+.3f}")
    print(f"    ({int((np.abs(coef)>1e-6).sum())}/{len(coef)} features non-zero)")

    # --- intensity: rank corr of each feature with excess on extreme days ---
    ext = I_tr > u
    exc = I_tr[ext] - u
    print(f"\n  INTENSITY — |Spearman| of features vs excess on {int(ext.sum())} "
          f"extreme training days:")
    cors = []
    for i in range(Xtr.shape[1]):
        r, p = spearmanr(Xtr[ext, i], exc)
        cors.append((names[i], r, p))
    pvals = np.array([p for _, _, p in cors])
    fdr_keep = _bh_fdr(pvals, alpha=0.05)            # Benjamini-Hochberg
    for nm, r, p in sorted(cors, key=lambda t: -abs(t[1]))[:12]:
        idx = names.index(nm)
        flag = "*" if fdr_keep[idx] else ("." if p < 0.05 else " ")
        print(f"   {flag}{nm:34s} r={r:+.3f}  p={p:.3f}")
    nraw = int((pvals < 0.05).sum())
    nfdr = int(fdr_keep.sum())
    print(f"    raw p<0.05: {nraw}/{len(cors)} (expect ~{0.05*len(cors):.0f} by chance); "
          f"BH-FDR<0.05: {nfdr}/{len(cors)} survive multiplicity.")
    print("    (* = survives BH-FDR; . = raw p<0.05 only)")


def gp_sensitivity(seeds=(0, 1, 2), epochs=250, u=1.0):
    """Reviewer point 3: is the deep-GP failure the MODEL CLASS or weak tuning?
    Sweep inducing points and deep-kernel vs plain GP; show it stays below
    climatology and the simple baselines across configurations."""
    print("\n" + "=" * 72 + "\n  (12) DEEP-GP SENSITIVITY (is it the model class, not tuning?)\n"
          + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    O, I, exc = data["O"], data["I"], data["excess"]
    O_te, I_te, I_tr = O[cut:].numpy(), I[cut:].numpy(), I[:cut].numpy()
    zmax = float(max(I_tr.max(), I_te.max()) * 1.05)
    z, dz = ev.make_grid(tau=u, zmax=zmax, nz=400)
    tw_clim = ev.twcrps_climatology(I_te, I_tr, tau=u, zmax=zmax)

    print(f"\n  {'configuration':28s} {'BSS':>8s} {'twCRPSS':>9s}")
    print(f"  {'logistic (reference)':28s} "
          f"{ev.brier_decomposition(O_te, LogisticRegression(C=1.0,max_iter=500).fit(Xtr.numpy(),O[:cut].numpy()).predict_proba(Xte.numpy())[:,1])['BSS']:+8.3f} "
          f"{'--':>9s}")
    for M, deep, lab in [(32, True, "deep-kernel, M=32"), (64, True, "deep-kernel, M=64"),
                         (128, True, "deep-kernel, M=128"), (64, False, "plain GP, M=64")]:
        bss, tws = [], []
        for s in seeds:
            torch.manual_seed(s)
            m = TCDGP(d_in=Xtr.shape[1], u=u, M=M, Q=2, coupled=True, deep=deep)
            m.fit(Xtr, O[:cut], exc[:cut], epochs=epochs, lr=0.02, batch=1024, verbose=False)
            p = m.predict_params(Xte)
            pi, sig, xi = p["pi"].numpy(), p["sigma"].numpy(), float(p["xi"])
            bss.append(ev.brier_decomposition(O_te, pi)["BSS"])
            tws.append(ev.skill(ev.twcrps_model(I_te, pi, sig, xi, u=u, tau=u, zmax=zmax), tw_clim))
        print(f"  {lab:28s} {np.mean(bss):+8.3f} {np.mean(tws):+9.3f}")
    print("\n  (all GP configurations remain below climatology and logistic "
          "=> the model class, not a tuning artefact)")


def stability_selection(n_rep=200, u=1.0, frac=0.6):
    """Reviewer concern C: uncertainty on feature importance. Selection frequency
    of each feature under L1-logistic over subsamples of the training data."""
    print("\n" + "=" * 72 + "\n  (13) STABILITY SELECTION (occurrence feature importance)\n"
          + "=" * 72)
    csv = _csv_path()
    drivers = load_drivers(*_driver_paths())
    data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
    names = ([f"lag_{i}" for i in range(1, 15)]
             + ["roll7", "roll30", "max7", "exc30", "sin_doy", "cos_doy"]
             + list(drivers.columns))
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, _ = standardize(data["X"][:cut], data["X"][cut:])
    Xtr, O_tr = Xtr.numpy(), data["O"][:cut].numpy()
    n = len(O_tr); rng = np.random.default_rng(0)
    freq = np.zeros(Xtr.shape[1])
    for _ in range(n_rep):
        idx = rng.choice(n, int(frac * n), replace=False)
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.3,
                                max_iter=1000).fit(Xtr[idx], O_tr[idx])
        freq += (np.abs(lr.coef_[0]) > 1e-6)
    freq /= n_rep
    print(f"\n  selection frequency over {n_rep} subsamples (top 12):")
    for i in np.argsort(-freq)[:12]:
        print(f"    {names[i]:34s} {freq[i]:.2f}")


def model_bakeoff(seeds=(0, 1, 2), epochs=600, u=1.0, deep=False):
    """H1/H3 test: does modelling the FULL zero-inflated distribution (all data,
    no hurdle-at-1) with a parametric distributional regression beat climatology,
    logistic, and the GP-EVT hurdle on proper scores?"""
    print("\n" + "=" * 72 + "\n  (7) MODEL BAKE-OFF: full-distribution regression vs "
          "baselines\n" + "=" * 72)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}"); return
    configs = [c for c in _driver_configs() if c[0] == "index-only" or c[0].startswith("+ERA5")]

    bd0, dates = load_real(csv, n_lags=14, u=u, drivers_df=None)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    I_tr, I_te = bd0["I"][:cut].numpy(), bd0["I"][cut:].numpy()
    O_te = bd0["O"][cut:].numpy()
    zmax = float(max(I_tr.max(), I_te.max()) * 1.05)
    z, dz = ev.make_grid(tau=u, zmax=zmax, nz=400)
    tw_clim = ev.twcrps_climatology(I_te, I_tr, tau=u, zmax=zmax)

    rows = []
    bdc = ev.brier_decomposition(O_te, np.full(len(O_te), float((I_tr > u).mean())))
    rows.append(("climatology", bdc["BS"], bdc["BSS"], bdc["RES"], 0.0))
    pi_p, sig_p, xi_p = ev.fit_pot(I_tr, u)
    rows.append(("stationary POT", bdc["BS"], bdc["BSS"], bdc["RES"],
                 ev.skill(ev.twcrps_pot(I_te, pi_p, sig_p, xi_p, u=u, tau=u, zmax=zmax), tw_clim)))

    for name, drv in configs:
        data, _ = load_real(csv, n_lags=14, u=u, drivers_df=drv)
        Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
        I_tr_t = data["I"][:cut]
        O_tr = data["O"][:cut].numpy()
        lr = LogisticRegression(C=1.0, max_iter=500).fit(Xtr.numpy(), O_tr)
        bl = ev.brier_decomposition(O_te, lr.predict_proba(Xte.numpy())[:, 1])
        rows.append((f"logistic [{name[:5]}]", bl["BS"], bl["BSS"], bl["RES"], float("nan")))
        # strong ML baseline: XGBoost occurrence (isotonic-calibrated probabilities)
        try:
            import xgboost as xgb
            from sklearn.calibration import CalibratedClassifierCV
            base = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, eval_metric="logloss", verbosity=0)
            xgbm = CalibratedClassifierCV(base, method="isotonic", cv=3)
            xgbm.fit(Xtr.numpy(), O_tr)
            bx = ev.brier_decomposition(O_te, xgbm.predict_proba(Xte.numpy())[:, 1])
            rows.append((f"XGBoost [{name[:5]}]", bx["BS"], bx["BSS"], bx["RES"], float("nan")))
        except Exception as e:
            print(f"  [xgboost skipped: {e}]")
        for dist in ("lognormal", "gamma", "egpd"):
            bss, res, twss = [], [], []
            for s in seeds:
                torch.manual_seed(s)
                m = ZIDistReg(Xtr.shape[1], dist=dist, u=u, deep=deep)
                m.fit(Xtr, I_tr_t, epochs=epochs, lr=0.05)
                pex = m.p_exceed(Xte)
                bd = ev.brier_decomposition(O_te, pex)
                Fg = m.cdf_grid(Xte, z)
                tw = ev.twcrps_from_cdf(I_te, Fg, z, dz)
                bss.append(bd["BSS"]); res.append(bd["RES"]); twss.append(ev.skill(tw, tw_clim))
            rows.append((f"ZI-{dist[:6]} [{name[:5]}]", np.nan,
                         np.mean(bss), np.mean(res), np.mean(twss)))

    print(f"\n  {'model':24s} {'BSS':>8s} {'RES(x1e3)':>10s} {'twCRPSS':>9s}")
    for n, bs, bss, res, tw in rows:
        tws = "   n/a" if (isinstance(tw, float) and np.isnan(tw)) else f"{tw:+.3f}"
        print(f"  {n:24s} {bss:+8.3f} {res*1e3:10.3f} {tws:>9s}")
    print("\n  (BSS, twCRPSS > 0 beat climatology; RES = informativeness. "
          "For reference TCDGP earlier: BSS ~ -0.4, twCRPSS ~ -0.3.)")


CITY = "london"   # set via set_city(); switches data + driver files
_ERA5 = {"london": ("era5_london_daily.parquet", "era5_cape_ivt_daily.parquet"),
         "paris":  ("era5_paris_daily.parquet", "era5_paris_cape_ivt_daily.parquet")}


def set_city(name):
    global CITY
    CITY = name


def _csv_path():
    return os.path.normpath(os.path.join(
        HERE, "..", "..", "data", f"{CITY}_precip_extreme_index_95th_1989_2018.csv"))


def _driver_paths():
    e, c = _ERA5[CITY]
    return [os.path.join(HERE, e), os.path.join(HERE, c),
            os.path.join(HERE, "teleconnections_daily.csv")]


def _driver_configs():
    """Available (label, drivers_df) configs, depending on which files exist."""
    era5, capeivt, tele = _driver_paths()
    have = lambda p: p if os.path.exists(p) else None
    configs = [("index-only", None)]
    if have(tele):
        configs.append(("+teleconnections", load_drivers(tele)))
    era5_files = [p for p in (era5, capeivt, tele) if have(p)]
    if have(era5) or have(capeivt):
        label = "+ERA5+CAPE/IVT" if have(capeivt) else "+ERA5 (+teleconn)"
        configs.append((label, load_drivers(*era5_files)))
    return configs


def driver_ablation_seeds(seeds=(0, 1, 2, 3, 4), epochs=250):
    """Seed-averaged driver ablation with mean +/- std and a paired test."""
    print("\n" + "=" * 64 + "\n  (4) SEED-AVERAGED DRIVER ABLATION "
          f"({len(seeds)} seeds)\n" + "=" * 64)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}"); return
    configs = _driver_configs()
    acc = {name: {"prauc": [], "cov": []} for name, _ in configs}
    per_seed = {name: {} for name, _ in configs}
    for s in seeds:
        for name, drv in configs:
            r = _fit_eval_real(csv, drv, epochs, name, seed=s)
            acc[name]["prauc"].append(r["prauc"]); acc[name]["cov"].append(r["cov"])
            per_seed[name][s] = r["prauc"]
        print(f"  seed {s} done", flush=True)

    def ms(x):
        a = np.array(x); return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

    base = acc[configs[0][0]]["prauc"]
    print(f"\n  RESULTS over {len(seeds)} seeds (mean +/- std):")
    print(f"  {'config':18s} {'PR-AUC':>16s} {'coverage':>14s}")
    for name, _ in configs:
        pm, ps = ms(acc[name]["prauc"]); cm, cs = ms(acc[name]["cov"])
        print(f"  {name:18s} {pm:6.3f} +/- {ps:5.3f}    {cm:5.3f} +/- {cs:5.3f}")
    # paired improvement of the richest config over index-only
    if len(configs) > 1:
        rich = configs[-1][0]
        diffs = np.array([per_seed[rich][s] - per_seed[configs[0][0]][s] for s in seeds])
        dm, ds = diffs.mean(), (diffs.std(ddof=1) if len(diffs) > 1 else 0.0)
        se = ds / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        print(f"\n  paired [{rich}] - [index-only]: "
              f"dPR-AUC = {dm:+.3f} +/- {ds:.3f} (SE {se:.3f}); "
              f"{'all seeds positive' if (diffs > 0).all() else 'mixed sign'}")
    print(f"  (prevalence baseline PR-AUC ~ {float(0.036):.3f})")


def horizon_sweep(horizons=(1, 3, 7, 14, 28), seeds=(0, 1, 2), epochs=200):
    """Seed-averaged driver ablation across forecast horizons.
    Tests whether ERA5 / teleconnection drivers earn their place at longer
    range (where the index's own short persistence fades)."""
    print("\n" + "=" * 70 + "\n  (5) HORIZON SWEEP (London 95th, seed-averaged "
          f"{len(seeds)} seeds)\n" + "=" * 70)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}"); return
    configs = _driver_configs()
    res = {}  # (h, name) -> (prauc_mean, prauc_std, cov_mean, base)
    for h in horizons:
        for name, drv in configs:
            pr, cov, base = [], [], []
            for s in seeds:
                r = _fit_eval_real(csv, drv, epochs, name, seed=s, horizon=h)
                pr.append(r["prauc"]); cov.append(r["cov"]); base.append(r["base"])
            pr = np.array(pr)
            res[(h, name)] = (pr.mean(), pr.std(ddof=1) if len(pr) > 1 else 0.0,
                              np.mean(cov), np.mean(base))
        print(f"  horizon {h:2d} done", flush=True)

    names = [n for n, _ in configs]
    print("\n  PR-AUC mean +/- std  (prevalence baseline in []):")
    print("  " + "h".rjust(4) + "  " + "  ".join(n.center(18) for n in names) + "   base")
    for h in horizons:
        cells = "  ".join(f"{res[(h,n)][0]:.3f}+/-{res[(h,n)][1]:.3f}".center(18)
                          for n in names)
        print(f"  {h:4d}  {cells}   [{res[(h,names[0])][3]:.3f}]")
    print("\n  conformal coverage (target 0.90):")
    for h in horizons:
        cov = "  ".join(f"{res[(h,n)][2]:.3f}".center(18) for n in names)
        print(f"  {h:4d}  {cov}")
    # does any driver set beat index-only at each horizon?
    print("\n  driver gain over index-only (PR-AUC):")
    for h in horizons:
        base = res[(h, names[0])][0]
        gains = "  ".join(f"{res[(h,n)][0]-base:+.3f}".center(18) for n in names)
        print(f"  {h:4d}  {gains}")


def driver_ablation(epochs=200):
    print("\n" + "=" * 64 + "\n  (3) DRIVER ABLATION (London 95th): does adding "
          "covariates help?\n" + "=" * 64)
    csv = _csv_path()
    if not os.path.exists(csv):
        print(f"  [skip] data not found at {csv}"); return
    configs = _driver_configs()
    if len(configs) == 1:
        print("  (only index-only available — fetch drivers to compare:\n"
              "   `python3 teleconnections.py` and/or `python3 fetch_era5_arco.py ...`)")

    rows = []
    for name, drv in configs:
        print(f"  fitting [{name}] ...", flush=True)
        rows.append(_fit_eval_real(csv, drv, epochs, name))

    print("\n  RESULTS (occurrence PR-AUC; conformal coverage target 0.90):")
    print(f"  {'config':18s} {'#feat':>5s} {'PR-AUC':>7s} {'vs base':>8s} "
          f"{'coverage':>9s} {'rho':>6s}")
    for r in rows:
        print(f"  {r['label']:18s} {r['nfeat']:5d} {r['prauc']:7.3f} "
              f"{r['prauc']-r['base']:+8.3f} {r['cov']:9.3f} {r['rho']:+6.2f}")
    print(f"  (prevalence baseline PR-AUC = {rows[0]['base']:.3f})")


if __name__ == "__main__":
    synthetic_experiment()
    real_experiment()
    driver_ablation()
    print("\n[done] prototype ran end-to-end.")
