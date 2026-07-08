"""Moving-block bootstrap CIs for the GP-EVT hurdle BSS and twCRPS (both cities)."""
import numpy as np, torch
import run_prototype as rp
from data import load_real, load_drivers, standardize
import evaluation as ev
from tcdgp import TCDGP

def block_boot(num, den, n_boot=2000, block=10, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([(np.arange(s, s+block) % n) for s in rng.integers(0, n, nb)])[:n]
        out[b] = 1 - num[idx].sum() / den[idx].sum()
    return out

for city in ("london", "paris"):
    rp.set_city(city)
    csv = rp._csv_path(); drivers = load_drivers(*rp._driver_paths())
    data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    O_te = data["O"][cut:].numpy(); I_tr = data["I"][:cut].numpy(); I_te = data["I"][cut:].numpy()
    zmax = float(max(I_tr.max(), I_te.max())*1.05)
    base = float((I_tr > 1.0).mean())
    # per-point climatology scores
    tw_clim_pt = ev.twcrps_climatology(I_te, I_tr, tau=1.0, zmax=zmax)
    br_clim_pt = (base - O_te)**2
    # seed-averaged GP per-point scores (600 epochs, matching Table 2)
    tw_gp, br_gp = [], []
    for s in (0,1,2):
        torch.manual_seed(s)
        m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=64, Q=2, coupled=True, deep=False)
        m.fit(Xtr, data["O"][:cut], data["excess"][:cut], epochs=600, lr=0.02, batch=1024, verbose=False)
        p = m.predict_params(Xte); pi=p["pi"].numpy(); sig=p["sigma"].numpy(); xi=float(p["xi"])
        tw_gp.append(ev.twcrps_model(I_te, pi, sig, xi, u=1.0, tau=1.0, zmax=zmax))
        br_gp.append((pi - O_te)**2)
    tw_gp_pt = np.mean(tw_gp, axis=0); br_gp_pt = np.mean(br_gp, axis=0)
    tw_skill = 1 - tw_gp_pt.sum()/tw_clim_pt.sum()
    br_skill = 1 - br_gp_pt.sum()/br_clim_pt.sum()
    tw_bs = block_boot(tw_gp_pt, tw_clim_pt); br_bs = block_boot(br_gp_pt, br_clim_pt)
    tl, th = np.percentile(tw_bs, [5,95]); bl, bh = np.percentile(br_bs, [5,95])
    print(f"== {city} (GP-EVT hurdle, seed-avg 3, 600ep) ==")
    print(f"  BSS    = {br_skill:+.3f}  90% CI [{bl:+.3f}, {bh:+.3f}]  P(>0)={np.mean(br_bs>0):.2f}")
    print(f"  twCRPS = {tw_skill:+.3f}  90% CI [{tl:+.3f}, {th:+.3f}]  P(>0)={np.mean(tw_bs>0):.2f}")
