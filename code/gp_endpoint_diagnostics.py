"""Check finite endpoints for every fitted model with a negative GPD shape."""
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_real
from major_revision_analysis import city_paths
from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import fit_penalised_hurdle_cv, seasonal_parameters


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
rows = []

for city in ("london", "paris"):
    tele, moisture, convective = corrected_driver_groups(city)
    drivers = pd.concat([tele, moisture, convective], axis=1)
    data, dates = load_real(city_paths(city)["index"], n_lags=14, u=1.0,
                            horizon=1, drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    index = data["I"][cut:].numpy()
    event = data["O"][cut:].numpy().astype(bool)
    excess = index[event] - 1.0
    seasonal = seasonal_parameters(data, dates, cut)
    linear = fit_penalised_hurdle_cv(
        data["X"][:cut], data["O"][:cut], data["excess"][:cut], data["X"][cut:]
    )
    gp = np.load(OUT / f"{city}_corrected_gp_main_predictions.npz")

    candidates = [
        ("seasonal POT", seasonal["sigma"], seasonal["xi"], "single fit"),
        ("regularised linear hurdle", linear["sigma"], linear["xi"], "single fit"),
    ]
    candidates.extend(
        ("GP-EVT hurdle", gp["sigma"][seed], float(xi), f"seed {seed}")
        for seed, xi in enumerate(gp["xi"])
    )
    for model, scale, xi, fit_id in candidates:
        if xi >= 0:
            continue
        scale = np.broadcast_to(np.asarray(scale), index.shape)
        endpoint = -scale[event] / xi
        ratio = excess / endpoint
        rows.append({
            "city": city, "model": model, "fit": fit_id, "xi": float(xi),
            "n_test_extremes": int(event.sum()),
            "n_at_or_beyond_endpoint": int((ratio >= 1).sum()),
            "maximum_excess_to_endpoint_ratio": float(ratio.max()),
            "minimum_endpoint_minus_excess": float((endpoint - excess).min()),
        })

result = pd.DataFrame(rows)
result.to_csv(OUT / "all_finite_endpoint_diagnostics.csv", index=False)
print(result.to_string(index=False))
