"""Paris supplementary figures (mirrors London Figs 1, 2, 4):
   fig_data_paris.pdf, fig_reliability_paris.pdf, fig_horizon_paris.pdf.

Inputs produced by prototype/_paris_supp.py:
   prototype/reliability_paris.npz, prototype/horizon_paris.json
and the Paris index CSV in data/.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.normpath(os.path.join(HERE, "..", "..", "prototype"))
CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "data", "paris_precip_extreme_index_95th_1989_2018.csv"))
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# ============================================================ Fig 1: EDA (Paris)
df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
I = df["index_max"]
ext = I > 1.0

fig, ax = plt.subplots(2, 2, figsize=(11, 6))
a = ax[0, 0]
a.plot(I.index, I.values, lw=0.3, color="steelblue", alpha=0.7)
a.scatter(I.index[ext], I.values[ext], s=6, color="crimson", zorder=5,
          label="extreme ($\\mathcal{I}>1$)")
a.axhline(1.0, color="k", ls="--", lw=0.7)
a.axvline(pd.Timestamp("2009-01-01"), color="grey", ls=":", lw=1)
a.text(pd.Timestamp("2009-06-01"), I.max() * 0.85, "test $\\rightarrow$",
       fontsize=8, color="grey")
a.set_ylabel("standardised index $\\mathcal{I}$")
a.set_title("(a) Daily index, 1989--2018 (train $|$ test split)")
a.legend(fontsize=8, frameon=False)

b = ax[0, 1]
b.hist(I.values, bins=60, color="#1b9e77", edgecolor="white", lw=0.3)
b.axvline(1.0, color="crimson", ls="--", lw=1.0, label="threshold $u=1$")
b.set_yscale("log")
b.set_xlabel("standardised index $\\mathcal{I}$")
b.set_ylabel("count (log scale)")
b.set_title("(b) Distribution: zero-inflated, right-skewed")
b.legend(fontsize=8, frameon=False)

c = ax[1, 0]
rate = ext.groupby(ext.index.month).mean() * 100
c.bar(range(1, 13), rate.values, color="#7570b3", alpha=0.85)
c.set_xticks(range(1, 13))
c.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
c.set_ylabel("extreme-day rate (\\%)")
c.set_title("(c) Seasonality: summer convective peak")

d = ax[1, 1]
x = ext.values.astype(float) - ext.mean()
n = len(x)
acf = np.array([1.0] + [np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, 31)])
d.bar(range(31), acf, color="#444444", width=0.7)
d.axhline(0, color="k", lw=0.6)
d.axhline(1.96 / np.sqrt(n), color="crimson", ls="--", lw=0.7)
d.axhline(-1.96 / np.sqrt(n), color="crimson", ls="--", lw=0.7)
d.set_xlabel("lag (days)")
d.set_ylabel("ACF of extreme indicator")
d.set_title("(d) Short memory: little persistence")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_data_paris.pdf"), bbox_inches="tight")
plt.close()
print("saved fig_data_paris.pdf")

# ===================================================== Fig 2: reliability (Paris)
d = np.load(os.path.join(PROTO, "reliability_paris.npz"), allow_pickle=True)
label_map = {"climatology": ("climatology", "crimson", "o"),
             "logistic+ERA5": ("logistic (+ERA5)", "#1b9e77", "s"),
             "TCDGP+ERA5": ("deep-GP--EVT (TCDGP)", "#7570b3", "^")}
fig, ax = plt.subplots(figsize=(5.0, 4.6))
ax.plot([0, 0.4], [0, 0.4], "k--", lw=0.8, label="perfect calibration")
xmax = 0.32
for k, (lab, col, mk) in label_map.items():
    if k not in d:
        continue
    xs, ys, ns = [np.asarray(a, dtype=float) for a in d[k]]
    ax.plot(xs, ys, mk + "-", color=col, ms=5, lw=1.3, label=lab)
    xmax = max(xmax, float(xs.max()) * 1.1 if xs.size else xmax)
ax.set_xlim(0, xmax); ax.set_ylim(0, xmax)
ax.set_xlabel("mean predicted exceedance probability")
ax.set_ylabel("observed exceedance frequency")
ax.set_title("Reliability diagram (Paris 95th, $h{=}1$)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_reliability_paris.pdf"), bbox_inches="tight")
plt.close()
print("saved fig_reliability_paris.pdf")

# ======================================================== Fig 4: horizon (Paris)
with open(os.path.join(PROTO, "horizon_paris.json")) as f:
    hz = json.load(f)
H = np.array(hz["H"])
PRAUC, COV, PREV = hz["PRAUC"], hz["COV"], hz["PREV"]
colors = {"index-only": "#444444", "+teleconnections": "#1b9e77", "+ERA5": "#7570b3"}
order = ["index-only", "+teleconnections", "+ERA5"]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for name in order:
    if name not in PRAUC:
        continue
    m, s = PRAUC[name]
    ax[0].errorbar(H, m, yerr=s, marker="o", ms=4, capsize=3, lw=1.4,
                   color=colors[name], label=name)
ax[0].axhline(PREV, ls="--", lw=0.9, color="crimson", label="prevalence")
ax[0].set_xscale("log"); ax[0].set_xticks(H); ax[0].set_xticklabels(H)
ax[0].set_xlabel("forecast horizon (days)"); ax[0].set_ylabel("occurrence PR-AUC")
ax[0].set_title("(a) Skill decays toward prevalence"); ax[0].legend(fontsize=8, frameon=False)

for name in order:
    if name not in COV:
        continue
    ax[1].plot(H, COV[name], marker="s", ms=4, lw=1.4, color=colors[name], label=name)
ax[1].axhline(0.90, ls="--", lw=0.9, color="black", label="nominal 0.90")
ax[1].set_xscale("log"); ax[1].set_xticks(H); ax[1].set_xticklabels(H)
ax[1].set_ylim(0.85, 0.95)
ax[1].set_xlabel("forecast horizon (days)"); ax[1].set_ylabel("conformal coverage")
ax[1].set_title("(b) Calibration holds at all horizons"); ax[1].legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_horizon_paris.pdf"), bbox_inches="tight")
plt.close()
print("saved fig_horizon_paris.pdf")
