"""Raw-data EDA figure (fig_data.pdf) for the London 95th-percentile index."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "data", "london_precip_extreme_index_95th_1989_2018.csv"))
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
I = df["index_max"]
ext = I > 1.0

fig, ax = plt.subplots(2, 2, figsize=(11, 6))

# (a) time series with extremes + train/test split
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

# (b) distribution (log count) — zero-inflation + right skew
b = ax[0, 1]
b.hist(I.values, bins=60, color="#1b9e77", edgecolor="white", lw=0.3)
b.axvline(1.0, color="crimson", ls="--", lw=1.0, label="threshold $u=1$")
b.set_yscale("log")
b.set_xlabel("standardised index $\\mathcal{I}$")
b.set_ylabel("count (log scale)")
b.set_title("(b) Distribution: zero-inflated, right-skewed")
b.legend(fontsize=8, frameon=False)

# (c) seasonality — monthly extreme rate
c = ax[1, 0]
rate = ext.groupby(ext.index.month).mean() * 100
c.bar(range(1, 13), rate.values, color="#7570b3", alpha=0.85)
c.set_xticks(range(1, 13))
c.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
c.set_ylabel("extreme-day rate (\\%)")
c.set_title("(c) Seasonality: summer convective peak")

# (d) autocorrelation of the extreme indicator
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
out = os.path.join(HERE, "fig_data.pdf")
plt.savefig(out, bbox_inches="tight")
print("saved", out)
