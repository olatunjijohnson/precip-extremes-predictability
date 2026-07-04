"""Horizon-decay + calibration figure from the seed-averaged sweep
(prototype/run_prototype.py::horizon_sweep, 3 seeds). Produces fig_horizon.pdf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = np.array([1, 3, 7, 14, 28])
PRAUC = {  # mean, std (GP--EVT hurdle, 3 seeds)
    "index-only":       ([0.076, 0.043, 0.044, 0.059, 0.052], [0.000, 0.001, 0.001, 0.003, 0.002]),
    "+teleconnections": ([0.071, 0.045, 0.046, 0.059, 0.052], [0.001, 0.001, 0.002, 0.002, 0.002]),
    "+ERA5":            ([0.118, 0.058, 0.049, 0.057, 0.055], [0.004, 0.002, 0.001, 0.003, 0.004]),
}
COV = {
    "index-only":       [0.901, 0.900, 0.900, 0.901, 0.900],
    "+teleconnections": [0.900, 0.900, 0.901, 0.901, 0.901],
    "+ERA5":            [0.900, 0.900, 0.900, 0.900, 0.900],
}
PREV = 0.036
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
colors = {"index-only": "#444444", "+teleconnections": "#1b9e77", "+ERA5": "#7570b3"}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for name, (m, s) in PRAUC.items():
    ax[0].errorbar(H, m, yerr=s, marker="o", ms=4, capsize=3, lw=1.4,
                   color=colors[name], label=name)
ax[0].axhline(PREV, ls="--", lw=0.9, color="crimson", label="prevalence")
ax[0].set_xscale("log"); ax[0].set_xticks(H); ax[0].set_xticklabels(H)
ax[0].set_xlabel("forecast horizon (days)"); ax[0].set_ylabel("occurrence PR-AUC")
ax[0].set_title("(a) Skill decays toward prevalence"); ax[0].legend(fontsize=8, frameon=False)

for name in COV:
    ax[1].plot(H, COV[name], marker="s", ms=4, lw=1.4, color=colors[name], label=name)
ax[1].axhline(0.90, ls="--", lw=0.9, color="black", label="nominal 0.90")
ax[1].set_xscale("log"); ax[1].set_xticks(H); ax[1].set_xticklabels(H)
ax[1].set_ylim(0.85, 0.95)
ax[1].set_xlabel("forecast horizon (days)"); ax[1].set_ylabel("conformal coverage")
ax[1].set_title("(b) Calibration holds at all horizons"); ax[1].legend(fontsize=8, frameon=False)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_horizon.pdf")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
