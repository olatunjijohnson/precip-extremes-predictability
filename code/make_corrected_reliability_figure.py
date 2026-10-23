"""CORP-style reliability diagnostics for the corrected 48-variable models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from data import load_real
from major_revision_analysis import city_paths
from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import (
    OUT, fit_penalised_hurdle_cv, moving_block_indices, seasonal_parameters,
)


def corp_terms(y, p):
    order = np.argsort(p)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrated = np.empty_like(p)
    calibrated[order] = iso.fit_transform(p[order], y[order])
    bs = np.mean((p-y)**2)
    bs_iso = np.mean((calibrated-y)**2)
    unc = np.mean((y-y.mean())**2)
    return bs, bs-bs_iso, unc-bs_iso, unc


def bootstrap_curve(y, p, grid, n_boot=500, block=14, seed=20261023):
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_boot):
        ii = moving_block_indices(len(y), rng, block)
        pp, yy = p[ii], y[ii]
        order = np.argsort(pp)
        if np.unique(yy).size < 2:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(pp[order], yy[order])
        curves.append(iso.predict(grid))
    return np.quantile(np.asarray(curves), [.025, .975], axis=0)


rows = []
curve_data = {}
for row, city in enumerate(("london", "paris")):
    tele, moisture, convective = corrected_driver_groups(city)
    drivers = pd.concat([tele, moisture, convective], axis=1)
    data, dates = load_real(city_paths(city)["index"], n_lags=14, u=1.0,
                            horizon=1, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    y = data["O"][cut:].numpy()
    seasonal = seasonal_parameters(data, dates, cut)
    linear = fit_penalised_hurdle_cv(data["X"][:cut], data["O"][:cut],
                                     data["excess"][:cut], data["X"][cut:])
    gp = np.load(OUT/f"{city}_corrected_gp_main_predictions.npz")
    forecasts = {"Penalised linear": linear["pi"],
                 "Gaussian process": gp["pi"].mean(0)}
    bs_ref = np.mean((seasonal["pi"]-y)**2)
    for col, (name, p) in enumerate(forecasts.items()):
        order = np.argsort(p)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(p[order], y[order])
        upper = max(.16, float(np.quantile(p, .995))*1.1)
        grid = np.linspace(max(0, float(p.min())), upper, 120)
        fitted = iso.predict(grid)
        lo, hi = bootstrap_curve(y, p, grid, seed=20261023+10*row+col)
        key = f"{city}_{'linear' if col == 0 else 'gp'}"
        curve_data[f"{key}_grid"] = grid
        curve_data[f"{key}_fitted"] = fitted
        curve_data[f"{key}_lo"] = lo
        curve_data[f"{key}_hi"] = hi
        bs, mcb, dsc, unc = corp_terms(y, p)
        pp = np.clip(p, 1e-12, 1-1e-12)
        rows.append({"city": city, "model": name, "mean_BS": bs,
                     "BSS_vs_seasonal": 1-bs/bs_ref,
                     "miscalibration": mcb, "discrimination": dsc,
                     "uncertainty": unc,
                     "mean_log_score": -np.mean(y*np.log(pp)+(1-y)*np.log(1-pp))})
np.savez_compressed(OUT/"corrected_reliability_curves.npz", **curve_data)
pd.DataFrame(rows).to_csv(OUT/"corrected_reliability_scores.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(OUT/"corrected_reliability_curves.npz")
