"""Independent and linear-kernel GP checks on the corrected 48 predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_real, standardize
from gp_review_sensitivity import fit_one
from major_revision_analysis import city_paths
from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import OUT, exact_twcrps_parts, seasonal_parameters


def analyse(city):
    tele, moisture, convective = corrected_driver_groups(city)
    drivers = pd.concat([tele, moisture, convective], axis=1)
    data, dates = load_real(city_paths(city)["index"], n_lags=14, u=1.0,
                            horizon=1, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    xtr, xte = standardize(data["X"][:cut], data["X"][cut:])
    index = data["I"][cut:].numpy()
    event = data["O"][cut:].numpy()
    seasonal = seasonal_parameters(data, dates, cut)
    ref = exact_twcrps_parts(index, seasonal["pi"], seasonal["sigma"],
                             seasonal["xi"])[0]
    bs_ref = np.mean((seasonal["pi"]-event)**2)
    rows = []
    for label, coupled, kernel in (
        ("rbf_independent_M64", False, "rbf"),
        ("linear_coupled_M64", True, "linear_bias"),
    ):
        for seed in (0, 1, 2):
            print(city, label, seed, flush=True)
            fit = fit_one(xtr, xte, data, cut, seed=seed, M=64,
                          coupled=coupled, kernel=kernel)
            score = exact_twcrps_parts(index, fit["pi"], fit["sigma"],
                                        fit["xi"])[0]
            rows.append({
                "city": city, "configuration": label, "seed": seed,
                "BSS_vs_seasonal": 1-np.mean((fit["pi"]-event)**2)/bs_ref,
                "twCRPSS_vs_seasonal": 1-score.mean()/ref.mean(),
                "xi": fit["xi"], "rho": fit["rho"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/f"{city}_corrected_gp_structure_sensitivity.csv", index=False)
    summary = frame.groupby("configuration").agg(
        BSS_mean=("BSS_vs_seasonal", "mean"), BSS_sd=("BSS_vs_seasonal", "std"),
        twCRPSS_mean=("twCRPSS_vs_seasonal", "mean"),
        twCRPSS_sd=("twCRPSS_vs_seasonal", "std"),
        xi_mean=("xi", "mean"), xi_sd=("xi", "std"),
        rho_mean=("rho", "mean"), rho_sd=("rho", "std"),
    ).reset_index()
    summary.to_csv(OUT/f"{city}_corrected_gp_structure_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name)
