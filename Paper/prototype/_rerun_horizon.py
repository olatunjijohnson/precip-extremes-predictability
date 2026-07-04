"""Regenerate horizon sweep (plain GP, deep=False) for London and Paris."""
import os, json, numpy as np
import run_prototype as rp
from data import load_real, standardize

HERE = os.path.dirname(os.path.abspath(__file__))
horizons = [1, 3, 7, 14, 28]
seeds = (0, 1, 2)

def sweep(city):
    rp.set_city(city)
    csv = rp._csv_path()
    configs = rp._driver_configs()
    label_for = {configs[0][0]: "index-only"}
    for name, _ in configs[1:]:
        label_for[name] = "+teleconnections" if ("teleconn" in name and "ERA5" not in name) else "+ERA5"
    prauc = {label_for[n]: ([], []) for n, _ in configs}
    cov = {label_for[n]: [] for n, _ in configs}
    prev = []
    for h in horizons:
        for name, drv in configs:
            lab = label_for[name]
            pr, cv, bs = [], [], []
            for s in seeds:
                r = rp._fit_eval_real(csv, drv, 200, lab, seed=s, horizon=h, deep=False)
                pr.append(r["prauc"]); cv.append(r["cov"]); bs.append(r["base"])
            prauc[lab][0].append(round(float(np.mean(pr)), 3))
            prauc[lab][1].append(round(float(np.std(pr, ddof=1)), 3))
            cov[lab].append(round(float(np.mean(cv)), 3))
            if lab == "index-only":
                prev.append(float(np.mean(bs)))
        print(f"  {city} horizon {h} done", flush=True)
    return dict(H=horizons, PRAUC=prauc, COV=cov, PREV=round(float(np.mean(prev)), 3))

for city in ("london", "paris"):
    out = sweep(city)
    fn = "horizon_london.json" if city == "london" else "horizon_paris.json"
    with open(os.path.join(HERE, fn), "w") as f:
        json.dump(out, f, indent=2)
    print(f"== {city} ==")
    print(json.dumps(out, indent=2))
