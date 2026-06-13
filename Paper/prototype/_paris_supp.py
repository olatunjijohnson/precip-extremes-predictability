"""Generate Paris supplementary-figure inputs:
   * reliability_paris.npz  (calibration curves)
   * horizon_paris.json     (seed-averaged horizon sweep: PR-AUC mean/std, coverage)
"""
import os, json, numpy as np
import run_prototype as r
from data import load_real, standardize

r.set_city("paris")
HERE = os.path.dirname(os.path.abspath(__file__))

# --- reliability curves for Paris (saves reliability_paris.npz) ---
r.calibration_eval(seeds=(0, 1, 2), epochs=250, save_npz="reliability_paris.npz")

# --- horizon sweep for Paris (collect numbers for the figure) ---
csv = r._csv_path()
configs = r._driver_configs()
# normalise labels to the three used in the London figure
label_for = {configs[0][0]: "index-only"}
for name, _ in configs[1:]:
    label_for[name] = "+teleconnections" if "teleconn" in name and "ERA5" not in name else "+ERA5"

horizons = [1, 3, 7, 14, 28]
seeds = (0, 1, 2)
prauc = {label_for[n]: ([], []) for n, _ in configs}
cov = {label_for[n]: [] for n, _ in configs}
prev = []
for h in horizons:
    for name, drv in configs:
        lab = label_for[name]
        pr, cv, bs = [], [], []
        for s in seeds:
            res = r._fit_eval_real(csv, drv, 200, lab, seed=s, horizon=h)
            pr.append(res["prauc"]); cv.append(res["cov"]); bs.append(res["base"])
        prauc[lab][0].append(float(np.mean(pr)))
        prauc[lab][1].append(float(np.std(pr, ddof=1)))
        cov[lab].append(float(np.mean(cv)))
        if lab == "index-only":
            prev.append(float(np.mean(bs)))
    print(f"  paris horizon {h} done", flush=True)

out = dict(H=horizons, PRAUC=prauc, COV=cov, PREV=float(np.mean(prev)))
with open(os.path.join(HERE, "horizon_paris.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved horizon_paris.json; prevalence ~", out["PREV"])
