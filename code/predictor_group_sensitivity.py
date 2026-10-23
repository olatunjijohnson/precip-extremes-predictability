"""Order and leave-one-group-out checks requested in the second review.

This is analysis-only: it writes no manuscript or Online Resource files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from second_review_analyses import (
    OUT, city_driver_groups, exact_twcrps_parts, fit_penalised_hurdle_cv,
    load_with_drivers, paired_interval, seasonal_parameters,
)


def corrected_driver_groups(city, extra_lag=1):
    """Move moisture transport out of the instability group and drop duplicate TCWV mean."""
    tele, era, mixed = city_driver_groups(city, extra_lag=extra_lag)
    instability_names = [
        "cape_mean", "cape_max", "cin_mean", "cin_max",
        "kx_mean", "kx_max", "totalx_mean", "totalx_max",
    ]
    moisture_names = [
        "tcwv_max", "viwve_mean", "viwve_max", "viwvn_mean",
        "viwvn_max", "ivt_mean",
    ]
    circulation_moisture = pd.concat([era, mixed[moisture_names]], axis=1)
    convective = mixed[instability_names]
    return tele, circulation_moisture, convective


def fit_spec(city, groups):
    tele, era, conv = corrected_driver_groups(city)
    lookup = {"teleconnections": tele, "circulation_moisture": era,
              "convective": conv}
    drivers = None if not groups else pd.concat([lookup[g] for g in groups], axis=1)
    if drivers is not None:
        drivers = drivers.loc[:, ~drivers.columns.duplicated()]
    data, dates, cut = load_with_drivers(city, drivers)
    fit = fit_penalised_hurdle_cv(data["X"][:cut], data["O"][:cut],
                                  data["excess"][:cut], data["X"][cut:])
    score, *_ = exact_twcrps_parts(data["I"][cut:].numpy(), fit["pi"],
                                   fit["sigma"], fit["xi"])
    return data, dates, cut, fit, score


def analyse(city):
    print(f"\nPREDICTOR-GROUP SENSITIVITY: {city.upper()}", flush=True)
    specs = {
        "history": (),
        "history + teleconnections": ("teleconnections",),
        "history + teleconnections + circulation/moisture":
            ("teleconnections", "circulation_moisture"),
        "full": ("teleconnections", "circulation_moisture", "convective"),
        "history + convective": ("convective",),
        "history + convective + circulation/moisture":
            ("convective", "circulation_moisture"),
        "history + teleconnections + convective":
            ("teleconnections", "convective"),
    }
    fits = {}
    for name, groups in specs.items():
        print(f"  fitting {name}", flush=True)
        fits[name] = fit_spec(city, groups)

    data, dates, cut, _, _ = fits["full"]
    seasonal = seasonal_parameters(data, dates, cut)
    reference, *_ = exact_twcrps_parts(data["I"][cut:].numpy(), seasonal["pi"],
                                        seasonal["sigma"], seasonal["xi"])

    score_rows = []
    for name, (_, _, _, fit, score) in fits.items():
        score_rows.append({
            "city": city, "model": name, "n_predictors": len(fit["occurrence_model"].coef_[0]),
            "mean_twCRPS": score.mean(),
            "twCRPSS_vs_seasonal": 1-score.mean()/reference.mean(),
            "lambda_occ": fit["lambda_occ"], "lambda_tail": fit["lambda_tail"],
            "xi": fit["xi"],
        })
    pd.DataFrame(score_rows).to_csv(
        OUT/f"{city}_corrected_predictor_group_scores.csv", index=False)

    contrasts = [
        ("published order: add teleconnections", "history + teleconnections", "history"),
        ("published order: add circulation/moisture",
         "history + teleconnections + circulation/moisture", "history + teleconnections"),
        ("published order: add convective", "full",
         "history + teleconnections + circulation/moisture"),
        ("reversed order: add convective after teleconnections",
         "history + teleconnections + convective", "history + teleconnections"),
        ("reversed order: add circulation/moisture",
         "full", "history + teleconnections + convective"),
        ("leave out teleconnections", "full",
         "history + convective + circulation/moisture"),
        ("leave out circulation/moisture", "full",
         "history + teleconnections + convective"),
        ("leave out convective", "full",
         "history + teleconnections + circulation/moisture"),
    ]
    rows = []
    for label, new_name, old_name in contrasts:
        new = fits[new_name][4]
        old = fits[old_name][4]
        lo, hi = paired_interval(new, old, reference, block=14,
                                 seed=20261023)
        rows.append({"city": city, "contrast": label,
                     "gain": (old.mean()-new.mean())/reference.mean(),
                     "ci_low": lo, "ci_high": hi})
    result = pd.DataFrame(rows)
    result.to_csv(OUT/f"{city}_corrected_predictor_group_contrasts.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name)
    (OUT/"corrected_predictor_group_settings.json").write_text(json.dumps({
        "ERA5_lag": "t-2", "bootstrap_blocks": "14 calendar days",
        "bootstrap_resamples": 2000, "interval": "percentile 95%",
        "penalty_selection": "three chronological validation folds at every stage",
        "group_correction": "TCWV and integrated vapour transport moved to circulation/moisture; near-duplicate tcwv_mean removed",
    }, indent=2))
