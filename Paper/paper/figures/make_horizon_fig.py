"""Combined horizon-decay + calibration figure for BOTH cities.
Top row London, bottom row Paris; left = occurrence PR-AUC, right = conformal
coverage. Reads prototype/horizon_london.json and horizon_paris.json
(seed-averaged GP--EVT hurdle sweep, 3 seeds). Produces fig_horizon.pdf."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.normpath(os.path.join(HERE, "..", "..", "prototype"))
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
colors = {"index-only": "#444444", "+teleconnections": "#1b9e77", "+ERA5": "#7570b3"}
order = ["index-only", "+teleconnections", "+ERA5"]

cities = [("London", "horizon_london.json"), ("Paris", "horizon_paris.json")]
fig, ax = plt.subplots(2, 2, figsize=(10, 7.0))
panel = iter("abcd")

for row, (city, fn) in enumerate(cities):
    hz = json.load(open(os.path.join(PROTO, fn)))
    H = np.array(hz["H"]); PRAUC, COV, PREV = hz["PRAUC"], hz["COV"], hz["PREV"]
    a = ax[row, 0]
    for name in order:
        if name not in PRAUC:
            continue
        m, s = PRAUC[name]
        a.errorbar(H, m, yerr=s, marker="o", ms=4, capsize=3, lw=1.4,
                   color=colors[name], label=name)
    a.axhline(PREV, ls="--", lw=0.9, color="crimson", label="prevalence")
    a.set_xscale("log"); a.set_xticks(H); a.set_xticklabels(H)
    a.set_xlabel("forecast horizon (days)"); a.set_ylabel("occurrence PR-AUC")
    a.set_title(f"({next(panel)}) {city}: skill decays toward prevalence")
    a.legend(fontsize=8, frameon=False)

    b = ax[row, 1]
    for name in order:
        if name not in COV:
            continue
        b.plot(H, COV[name], marker="s", ms=4, lw=1.4, color=colors[name], label=name)
    b.axhline(0.90, ls="--", lw=0.9, color="black", label="nominal 0.90")
    b.set_xscale("log"); b.set_xticks(H); b.set_xticklabels(H)
    b.set_ylim(0.85, 0.95)
    b.set_xlabel("forecast horizon (days)"); b.set_ylabel("conformal coverage")
    b.set_title(f"({next(panel)}) {city}: calibration holds at all horizons")
    b.legend(fontsize=8, frameon=False)

plt.tight_layout()
out = os.path.join(HERE, "fig_horizon.pdf")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
