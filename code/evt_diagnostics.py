"""Training-period threshold and stationary-GPD diagnostics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto

from major_revision_analysis import OUT, U, city_paths
from data import load_real


THRESHOLDS = np.arange(0.6, 1.41, 0.1)
N_BOOT = 200


def fit_shape(y):
    return float(genpareto.fit(y, floc=0)[0])


def diagnostics(city, rng):
    d, dates = load_real(city_paths(city)["index"], n_lags=14, u=U,
                         horizon=1, drivers_df=None)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    values = d["I"][:cut].numpy()
    rows = []
    for threshold in THRESHOLDS:
        excess = values[values > threshold] - threshold
        xi = fit_shape(excess)
        boot = np.array([fit_shape(rng.choice(excess, len(excess), replace=True))
                         for _ in range(N_BOOT)])
        rows.append({"city": city, "threshold": threshold,
                     "n_exceedances": len(excess),
                     "mean_excess": excess.mean(), "xi": xi,
                     "xi_ci_low": np.quantile(boot, 0.025),
                     "xi_ci_high": np.quantile(boot, 0.975)})
    return values, pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(20261015)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    all_rows = []
    for row, city in enumerate(("london", "paris")):
        values, tab = diagnostics(city, rng)
        all_rows.append(tab)
        axes[row, 0].plot(tab.threshold, tab.mean_excess, "o-")
        axes[row, 0].axvline(U, color="0.4", ls=":")
        axes[row, 0].set_ylabel(f"{city.title()}\nmean excess")
        axes[row, 0].set_xlabel("threshold")

        axes[row, 1].plot(tab.threshold, tab.xi, "o-")
        axes[row, 1].fill_between(tab.threshold, tab.xi_ci_low,
                                  tab.xi_ci_high, alpha=.2)
        axes[row, 1].axvline(U, color="0.4", ls=":")
        axes[row, 1].axhline(0, color="0.7", lw=.8)
        axes[row, 1].set_ylabel("GPD shape")
        axes[row, 1].set_xlabel("threshold")

        excess = np.sort(values[values > U] - U)
        xi, _, sigma = genpareto.fit(excess, floc=0)
        prob = (np.arange(1, len(excess)+1)-.5)/len(excess)
        theo = genpareto.ppf(prob, xi, loc=0, scale=sigma)
        lim = max(excess.max(), theo.max())
        axes[row, 2].scatter(theo, excess, s=10, alpha=.65)
        axes[row, 2].plot([0, lim], [0, lim], color="0.4", ls=":")
        axes[row, 2].set_xlabel("fitted GPD quantile")
        axes[row, 2].set_ylabel("observed excess quantile")

    axes[0, 0].set_title("Mean residual life")
    axes[0, 1].set_title("Parameter stability")
    axes[0, 2].set_title("Stationary-GPD QQ plot")
    fig.tight_layout()
    fig.savefig(OUT / "evt_diagnostics.pdf", bbox_inches="tight")
    pd.concat(all_rows).to_csv(OUT / "evt_threshold_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
