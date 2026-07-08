"""Does the GP-EVT hurdle still win the tail WITHOUT coupling (independent)?"""
import numpy as np, torch
import run_prototype as rp
from data import load_real, load_drivers, standardize
import evaluation as ev
from tcdgp import TCDGP
for city in ("london","paris"):
    rp.set_city(city)
    csv = rp._csv_path(); drivers = load_drivers(*rp._driver_paths())
    data, dates = load_real(csv, n_lags=14, u=1.0, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    Xtr, Xte = standardize(data["X"][:cut], data["X"][cut:])
    O_te = data["O"][cut:].numpy(); I_tr = data["I"][:cut].numpy(); I_te = data["I"][cut:].numpy()
    zmax = float(max(I_tr.max(), I_te.max())*1.05)
    tw_clim = ev.twcrps_climatology(I_te, I_tr, tau=1.0, zmax=zmax)
    for coupled in (True, False):
        bss, tws = [], []
        for s in (0,1,2):
            torch.manual_seed(s)
            m = TCDGP(d_in=Xtr.shape[1], u=1.0, M=64, Q=2, coupled=coupled, deep=False)
            m.fit(Xtr, data["O"][:cut], data["excess"][:cut], epochs=600, lr=0.02, batch=1024, verbose=False)
            p = m.predict_params(Xte); pi=p["pi"].numpy(); sig=p["sigma"].numpy(); xi=float(p["xi"])
            bss.append(ev.brier_decomposition(O_te, pi)["BSS"])
            tws.append(ev.skill(ev.twcrps_model(I_te, pi, sig, xi, u=1.0, tau=1.0, zmax=zmax), tw_clim))
        tag = "coupled  " if coupled else "independent"
        print(f"  {city:7s} {tag}  BSS={np.mean(bss):+.3f}  twCRPS={np.mean(tws):+.3f}")
