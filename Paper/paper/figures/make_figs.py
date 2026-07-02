"""Generate fig_reliability.pdf and fig_intensity.pdf for the manuscript."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.normpath(os.path.join(HERE, "..", "..", "prototype"))
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# ---------------------------------------------------------------- reliability
d = np.load(os.path.join(PROTO, "reliability.npz"), allow_pickle=True)
label_map = {"climatology": ("climatology", "crimson", "o"),
             "logistic+ERA5": ("logistic (+ERA5)", "#1b9e77", "s"),
             "TCDGP+ERA5": ("deep-GP--EVT (TCDGP)", "#7570b3", "^")}
fig, ax = plt.subplots(figsize=(5.0, 4.6))
ax.plot([0, 0.35], [0, 0.35], "k--", lw=0.8, label="perfect calibration")
for k, (lab, col, mk) in label_map.items():
    if k not in d:
        continue
    xs, ys, ns = [np.asarray(a, dtype=float) for a in d[k]]
    ax.plot(xs, ys, mk + "-", color=col, ms=5, lw=1.3, label=lab)
ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.32)
ax.set_xlabel("mean predicted exceedance probability")
ax.set_ylabel("observed exceedance frequency")
ax.set_title("Reliability diagram (London 95th, $h{=}1$)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_reliability.pdf"), bbox_inches="tight")
plt.close()
print("saved fig_reliability.pdf")

# ---------------------------------------------------------------- intensity forest (two cities)
periods = ["2003--2018", "2006--2018", "2009--2018", "2012--2018"]
lon = dict(skill=np.array([0.037, 0.027, 0.035, 0.034]),
           lo=np.array([0.020, 0.015, 0.021, 0.006]),
           hi=np.array([0.051, 0.039, 0.050, 0.059]))
par = dict(skill=np.array([0.043, 0.043, 0.035, 0.046]),
           lo=np.array([0.019, 0.019, 0.008, 0.019]),
           hi=np.array([0.064, 0.064, 0.061, 0.073]))
base = np.arange(len(periods))[::-1]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for d, off, col, lab in [(lon, +0.12, "#444444", "London"),
                         (par, -0.12, "#7570b3", "Paris")]:
    ax.errorbar(d["skill"], base + off, xerr=[d["skill"] - d["lo"], d["hi"] - d["skill"]],
                fmt="o", color=col, ms=6, capsize=4, lw=1.5, label=lab)
ax.axvline(0, color="crimson", ls="--", lw=1.0)
ax.set_yticks(base); ax.set_yticklabels(periods)
ax.set_xlim(-0.01, 0.085)
ax.set_xlabel("CRPS skill (excess) vs. climatology")
ax.set_ylabel("test period")
ax.set_title("Hold-out intensity predictability (90\\% moving-block bootstrap CI)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_intensity.pdf"), bbox_inches="tight")
plt.close()
print("saved fig_intensity.pdf")
