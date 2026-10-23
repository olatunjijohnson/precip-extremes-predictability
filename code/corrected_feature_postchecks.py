"""Threshold and bootstrap checks for the final corrected 48-variable fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_real
from major_revision_analysis import city_paths
from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import (
    N_BOOT, OUT, conditional_calendar_interval, exact_twcrps_parts,
    fit_penalised_hurdle_cv, moving_block_indices, paired_interval,
    seasonal_parameters,
)


def analyse(city):
    tele, moisture, convective = corrected_driver_groups(city)
    drivers = pd.concat([tele, moisture, convective], axis=1)
    paths = city_paths(city)
    data, dates = load_real(paths["index"], n_lags=14, u=1.0, horizon=1,
                            drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    index = data["I"][cut:].numpy()
    event = data["O"][cut:].numpy()
    seasonal = seasonal_parameters(data, dates, cut)
    linear = fit_penalised_hurdle_cv(data["X"][:cut], data["O"][:cut],
                                     data["excess"][:cut], data["X"][cut:])
    gp = np.load(OUT/f"{city}_corrected_gp_main_predictions.npz")

    threshold_rows = []
    for tau in (1.0, 1.25, 1.5, 2.0):
        ref = exact_twcrps_parts(index, seasonal["pi"], seasonal["sigma"],
                                 seasonal["xi"], tau=tau)[0]
        lin = exact_twcrps_parts(index, linear["pi"], linear["sigma"],
                                 linear["xi"], tau=tau)[0]
        gps = np.mean([exact_twcrps_parts(index, gp["pi"][s], gp["sigma"][s],
                                          float(gp["xi"][s]), tau=tau)[0]
                       for s in range(5)], axis=0)
        threshold_rows.append({
            "city": city, "tau": tau, "n_test_above_tau": int((index > tau).sum()),
            "mean_reference": ref.mean(), "mean_linear": lin.mean(),
            "mean_gp": gps.mean(), "linear_skill": 1-lin.mean()/ref.mean(),
            "gp_skill": 1-gps.mean()/ref.mean(),
            "gp_minus_linear": (lin.mean()-gps.mean())/ref.mean(),
        })
    pd.DataFrame(threshold_rows).to_csv(
        OUT/f"{city}_corrected_verification_thresholds.csv", index=False)

    ref = exact_twcrps_parts(index, seasonal["pi"], seasonal["sigma"],
                             seasonal["xi"])[0]
    lin = exact_twcrps_parts(index, linear["pi"], linear["sigma"],
                             linear["xi"])[0]
    gps = np.mean([exact_twcrps_parts(index, gp["pi"][s], gp["sigma"][s],
                                      float(gp["xi"][s]))[0]
                   for s in range(5)], axis=0)
    rows = []
    for block in (7, 14, 30, 90):
        lo, hi = paired_interval(gps, lin, ref, block=block,
                                 seed=20261023+block)
        rows.append({"city": city, "comparison": "GP minus linear",
                     "block": f"{block} calendar days",
                     "estimate": (lin.mean()-gps.mean())/ref.mean(),
                     "ci_low": lo, "ci_high": hi})

    ext = event.astype(bool)
    y = index[ext]-1.0
    cond_lin = exact_twcrps_parts(y, np.ones(ext.sum()), linear["sigma"][ext],
                                  linear["xi"], tau=0.0, u=0.0)[0]
    cond_gp = np.mean([exact_twcrps_parts(
        y, np.ones(ext.sum()), gp["sigma"][s, ext], float(gp["xi"][s]),
        tau=0.0, u=0.0)[0] for s in range(5)], axis=0)
    rng = np.random.default_rng(20261023)
    vals = []
    for _ in range(N_BOOT):
        ii = moving_block_indices(ext.sum(), rng, 10)
        vals.append(np.mean(cond_lin[ii]-cond_gp[ii]))
    lo, hi = np.quantile(vals, [.025, .975])
    rows.append({"city": city, "comparison": "conditional GP minus linear",
                 "block": "10 extreme observations",
                 "estimate": np.mean(cond_lin-cond_gp), "ci_low": lo, "ci_high": hi})
    for block in (14, 30, 90):
        lo, hi = conditional_calendar_interval(cond_gp, cond_lin, ext, block,
                                                seed=20261023+block)
        rows.append({"city": city, "comparison": "conditional GP minus linear",
                     "block": f"{block} calendar days",
                     "estimate": np.mean(cond_lin-cond_gp),
                     "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(OUT/f"{city}_corrected_bootstrap_sensitivity.csv",
                              index=False)
    print(pd.DataFrame(threshold_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name)
