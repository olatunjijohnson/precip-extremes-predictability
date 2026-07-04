"""
revision_analyses.py — reproduces the additional analyses added to the paper
during revision (in response to reviewer comments). Each section prints the
numbers quoted in the corresponding place in the manuscript.

Sections
  (A) Sensitivity to the exceedance threshold u   -> Sec 5.1 "Sensitivity to u"
  (B) Bootstrap block-size sensitivity            -> Sec 4.4 "Robustness"
  (C) Synthetic dependent-hurdle misspecification -> Sec 5.5 "Misspecification"
  (D) Real-data occurrence-intensity correlation  -> Sec 5.5 "What does the
                                                     real data say?"
  (E) Feature-importance table (both cities)      -> Supplementary Table S2
  (F) Horizon-wise occurrence BSS (both cities)   -> Supplementary Table S3
  (G) Bake-off at the 99th-percentile threshold   -> Supplementary Table S1

The GP sections (C, D) are the slow ones (a few minutes each); the distributional
models in (G) also take a couple of minutes; the rest run in well under a minute.
Run:  python3 revision_analyses.py [A|B|C|D|E|F|G|all]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

import run_prototype as rp
import evaluation as ev
from data import make_synthetic, load_real, load_drivers, standardize
from tcdgp import TCDGP
from sklearn.linear_model import LogisticRegression, QuantileRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = ("london", "paris")


def _pinball(y, P, qs):
    """Elementwise pinball loss, shape (n, nq)."""
    e = y[:, None] - P
    return np.maximum(qs[None, :] * e, (qs[None, :] - 1) * e)


# ---------------------------------------------------------------------------
def section_A_threshold_sensitivity():
    """Logistic (+ERA5) occurrence Brier skill at u in {0.75, 1.0, 1.25}."""
    print("\n(A) THRESHOLD SENSITIVITY  (logistic +ERA5 BSS vs climatology)")
    for city in CITIES:
        rp.set_city(city)
        csv = rp._csv_path()
        drivers = load_drivers(*rp._driver_paths())
        out = []
        for u in (0.75, 1.00, 1.25):
            data, dates = load_real(csv, n_lags=14, u=u, drivers_df=drivers)
            cut = int((dates <= np.datetime64("2008-12-31")).sum())
            Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
            O_tr, O_te = data["O"][:cut].numpy(), data["O"][cut:].numpy()
            lr = LogisticRegression(C=1.0, max_iter=500).fit(Xtr.numpy(), O_tr)
            bss = ev.brier_decomposition(O_te, lr.predict_proba(Xte.numpy())[:, 1])["BSS"]
            out.append(bss)
        print(f"  {city:7s}  u=0.75:{out[0]:+.3f}  u=1.00:{out[1]:+.3f}  u=1.25:{out[2]:+.3f}")


# ---------------------------------------------------------------------------
def section_B_blocksize_sensitivity():
    """CRPS-of-excess intensity CI on the SMALLEST split (test 2012-2018),
    across moving-block bootstrap block lengths {5, 10, 20}."""
    print("\n(B) BOOTSTRAP BLOCK-SIZE SENSITIVITY  (smallest split, test 2012-2018)")
    qs = np.arange(0.1, 0.91, 0.1)
    for city in CITIES:
        rp.set_city(city)
        csv = rp._csv_path()
        drivers = load_drivers(*rp._driver_paths())
        data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
        X = data["X"]; I = data["I"].numpy(); yr = dates.year.values
        tr, te = yr <= 2011, yr > 2011
        Xtr, Xte = standardize(X[tr], X[te]); Xtr, Xte = Xtr.numpy(), Xte.numpy()
        Itr, Ite = I[tr], I[te]
        etr, ete = Itr > 1.0, Ite > 1.0
        Xetr, exc_tr = Xtr[etr], Itr[etr] - 1.0
        Xete, exc_te = Xte[ete], Ite[ete] - 1.0
        preds = np.zeros((int(ete.sum()), len(qs)))
        for j, q in enumerate(qs):
            m = QuantileRegressor(quantile=q, alpha=0.05, solver="highs").fit(Xetr, exc_tr)
            preds[:, j] = np.clip(m.predict(Xete), 0, None)
        preds = np.sort(preds, axis=1)
        clim = np.quantile(exc_tr, qs)
        cm = 2 * _pinball(exc_te, preds, qs).mean(1)
        cc = 2 * _pinball(exc_te, np.broadcast_to(clim, preds.shape), qs).mean(1)
        print(f"  {city:7s}  n_test_extreme={int(ete.sum())}  point skill={1 - cm.sum() / cc.sum():+.3f}")
        for block in (5, 10, 20):
            bs = rp._block_boot(cm, cc, n_boot=2000, block=block, seed=0)
            lo, hi = np.percentile(bs, [5, 95])
            print(f"           block={block:2d}: 90% CI=[{lo:+.3f}, {hi:+.3f}]  P>0={np.mean(bs > 0):.2f}")


# ---------------------------------------------------------------------------
def _tail_loglik(model, X, O, excess):
    mask = O == 1
    if mask.sum() == 0:
        return float("nan")
    p = model.predict_params(X)
    ll = model.lik._gpd_logpdf(excess[mask].clamp_min(0), p["sigma"][mask], p["xi"])
    return float(ll.mean())


def section_C_misspecification(seeds=(1, 2, 3)):
    """Synthetic dependent-hurdle recovery at rho_true in {0.3, 0.7, 0.9}."""
    print("\n(C) SYNTHETIC MISSPECIFICATION  (coupled-hurdle rho recovery)")
    for rho_true in (0.3, 0.7, 0.9):
        rhat, gain = [], []
        for seed in seeds:
            data, _ = make_synthetic(n=4000, rho=rho_true, xi=0.2, base_rate=0.08, seed=seed)
            ntr = 3000
            Xtr, Xte = standardize(data["X"][:ntr], data["X"][ntr:])
            O_te, exc_te = data["O"][ntr:], data["excess"][ntr:]
            tlls = {}
            for coupled in (False, True):
                torch.manual_seed(0)
                m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=48, Q=2, coupled=coupled, deep=False)
                m.fit(Xtr, data["O"][:ntr], data["excess"][:ntr], epochs=300, lr=0.02, verbose=False)
                tlls[coupled] = _tail_loglik(m, Xte, O_te, exc_te)
                if coupled:
                    rhat.append(float(m.gp.rho()))
            gain.append(tlls[True] - tlls[False])
        rhat = np.array(rhat)
        print(f"  rho_true={rho_true}:  rho_hat={rhat.mean():+.3f} +/- {rhat.std():.3f}"
              f"   tailLL gain mean={np.mean(gain):+.1f}")


def section_D_real_rho(seeds=range(5)):
    """Coupled-hurdle rho_hat on the real London/Paris series (multi-seed)."""
    print("\n(D) REAL-DATA rho_hat  (coupled GP-EVT hurdle, multi-seed)")
    for city in CITIES:
        rp.set_city(city)
        csv = rp._csv_path()
        drivers = load_drivers(*rp._driver_paths())
        data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
        cut = int((dates <= np.datetime64("2008-12-31")).sum())
        Xtr, _ = standardize(data["X"][:cut], data["X"][cut:])
        rhos = []
        for seed in seeds:
            torch.manual_seed(seed)
            m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=64, Q=2, coupled=True, deep=False)
            m.fit(Xtr, data["O"][:cut], data["excess"][:cut], epochs=250, lr=0.02,
                  batch=1024, verbose=False)
            rhos.append(float(m.gp.rho()))
        rhos = np.array(rhos)
        print(f"  {city:7s}  rho_hat={rhos.mean():+.3f} +/- {rhos.std():.3f}  "
              f"(per seed: {[f'{r:+.3f}' for r in rhos]})")


# ---------------------------------------------------------------------------
def _feature_names(drivers):
    return ([f"lag_{i}" for i in range(1, 15)]
            + ["roll7", "roll30", "max7", "exc30", "sin_doy", "cos_doy"]
            + list(drivers.columns))


def section_E_feature_importance(n_rep=200, frac=0.6, top=15):
    """L1-logistic stability-selection frequency, top features, both cities."""
    print("\n(E) FEATURE IMPORTANCE  (L1-logistic stability selection, top 15 of d=49)")
    for city in CITIES:
        rp.set_city(city)
        csv = rp._csv_path()
        drivers = load_drivers(*rp._driver_paths())
        data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
        names = _feature_names(drivers)
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
        print(f"  {city} (d={Xtr.shape[1]}):")
        for i in np.argsort(-freq)[:top]:
            print(f"     {names[i]:30s} {freq[i]:.2f}")


def section_F_horizon_bss():
    """Logistic (+ERA5) occurrence BSS and resolution by horizon, both cities."""
    print("\n(F) HORIZON-WISE OCCURRENCE BSS  (logistic +ERA5)")
    for city in CITIES:
        rp.set_city(city)
        csv = rp._csv_path()
        drivers = load_drivers(*rp._driver_paths())
        print(f"  {city}:")
        for h in (1, 3, 7, 14, 28):
            data, dates = load_real(csv, n_lags=14, u=1.0, horizon=h, drivers_df=drivers)
            cut = int((dates <= np.datetime64("2008-12-31")).sum())
            Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
            O_tr, O_te = data["O"][:cut].numpy(), data["O"][cut:].numpy()
            lr = LogisticRegression(C=1.0, max_iter=500).fit(Xtr.numpy(), O_tr)
            bd = ev.brier_decomposition(O_te, lr.predict_proba(Xte.numpy())[:, 1])
            print(f"     h={h:2d}: BSS={bd['BSS']:+.3f}  RES(x1e3)={bd['RES'] * 1e3:.3f}")


def section_G_bakeoff_99th(seeds=(0, 1, 2), epochs=600):
    """London bake-off at the 99th-percentile threshold (Supplementary Table S1).
    Reuses run_prototype.model_bakeoff by pointing it at the 99th index file."""
    print("\n(G) BAKE-OFF AT THE 99th PERCENTILE  (London; cf. Supplementary Table S1)")
    rp.set_city("london")
    csv99 = os.path.normpath(os.path.join(
        HERE, "..", "..", "data", "london_precip_extreme_index_99th_1989_2018.csv"))
    if not os.path.exists(csv99):
        print(f"  [skip] 99th index not found at {csv99}")
        return
    orig = rp._csv_path
    rp._csv_path = lambda: csv99            # redirect the bake-off to the 99th index
    try:
        rp.model_bakeoff(seeds=seeds, epochs=epochs, u=1.0)
    finally:
        rp._csv_path = orig                 # restore, so later sections use the 95th


SECTIONS = {"A": section_A_threshold_sensitivity, "B": section_B_blocksize_sensitivity,
            "C": section_C_misspecification, "D": section_D_real_rho,
            "E": section_E_feature_importance, "F": section_F_horizon_bss,
            "G": section_G_bakeoff_99th}

if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    keys = list(SECTIONS) if which == "ALL" else list(which)
    for k in keys:
        SECTIONS[k]()
    print("\ndone.")
