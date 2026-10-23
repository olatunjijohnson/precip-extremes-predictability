"""Primary GP/linear comparison after correcting the 48-variable feature set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_real, standardize
from gp_review_sensitivity import MAIN_SEEDS, fit_one
from major_revision_analysis import city_paths
from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import (
    OUT, exact_twcrps_parts, fit_penalised_hurdle_cv, paired_interval,
    seasonal_parameters,
)


def analyse(city):
    tele, moisture, convective = corrected_driver_groups(city)
    drivers = pd.concat([tele, moisture, convective], axis=1)
    paths = city_paths(city)
    data, dates = load_real(paths["index"], n_lags=14, u=1.0, horizon=1,
                            drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    xtr, xte = standardize(data["X"][:cut], data["X"][cut:])
    index = data["I"][cut:].numpy()
    event = data["O"][cut:].numpy()
    seasonal = seasonal_parameters(data, dates, cut)
    reference, *_ = exact_twcrps_parts(index, seasonal["pi"], seasonal["sigma"],
                                        seasonal["xi"])

    print(f"\nCORRECTED 48-VARIABLE GP: {city.upper()}", flush=True)
    fitted = []
    gp_scores = []
    rows = []
    for seed in MAIN_SEEDS:
        print(f"  seed {seed}", flush=True)
        result = fit_one(xtr, xte, data, cut, seed=seed, M=64,
                         coupled=True, kernel="rbf")
        score, *_ = exact_twcrps_parts(index, result["pi"], result["sigma"],
                                       result["xi"])
        fitted.append(result)
        gp_scores.append(score)
        rows.append({
            "city": city, "seed": seed,
            "BSS_vs_seasonal": 1-np.mean((result["pi"]-event)**2)
                / np.mean((seasonal["pi"]-event)**2),
            "twCRPSS_vs_seasonal": 1-score.mean()/reference.mean(),
            "xi": result["xi"], "rho": result["rho"],
        })
    np.savez_compressed(
        OUT/f"{city}_corrected_gp_main_predictions.npz",
        pi=np.stack([r["pi"] for r in fitted]),
        sigma=np.stack([r["sigma"] for r in fitted]),
        xi=np.array([r["xi"] for r in fitted]),
        pi_draw=np.stack([r["pi_draw"] for r in fitted]),
        sigma_draw=np.stack([r["sigma_draw"] for r in fitted]),
        rho=np.array([r["rho"] for r in fitted]),
        dates=dates[cut:].to_numpy(dtype="datetime64[D]"),
        observed_index=index, observed_event=event,
    )
    pd.DataFrame(rows).to_csv(OUT/f"{city}_corrected_gp_seeds.csv", index=False)

    print("  fitting corrected penalised linear hurdle", flush=True)
    linear = fit_penalised_hurdle_cv(data["X"][:cut], data["O"][:cut],
                                     data["excess"][:cut], data["X"][cut:])
    linear_score, *_ = exact_twcrps_parts(index, linear["pi"], linear["sigma"],
                                           linear["xi"])
    gp_score = np.mean(gp_scores, axis=0)
    gp_pi = np.stack([r["pi"] for r in fitted])
    gp_sigma = np.stack([r["sigma"] for r in fitted])
    gp_xi = np.array([r["xi"] for r in fitted])

    specs = {
        "seasonal occurrence + seasonal intensity":
            [(seasonal["pi"], seasonal["sigma"], seasonal["xi"])],
        "linear occurrence + seasonal intensity":
            [(linear["pi"], seasonal["sigma"], seasonal["xi"])],
        "linear occurrence + linear intensity":
            [(linear["pi"], linear["sigma"], linear["xi"])],
        "GP occurrence + linear intensity":
            [(gp_pi[s], linear["sigma"], linear["xi"]) for s in range(5)],
        "linear occurrence + GP intensity":
            [(linear["pi"], gp_sigma[s], float(gp_xi[s])) for s in range(5)],
        "GP occurrence + GP intensity":
            [(gp_pi[s], gp_sigma[s], float(gp_xi[s])) for s in range(5)],
    }
    scores = {}
    score_rows = []
    for name, combinations in specs.items():
        score = np.mean([exact_twcrps_parts(index, *p)[0]
                         for p in combinations], axis=0)
        scores[name] = score
        score_rows.append({"city": city, "model": name,
                           "mean_twCRPS": score.mean(),
                           "twCRPSS_vs_seasonal": 1-score.mean()/reference.mean()})

    comparisons = [
        ("linear occurrence added to seasonal intensity",
         "linear occurrence + seasonal intensity",
         "seasonal occurrence + seasonal intensity"),
        ("linear intensity added with occurrence fixed",
         "linear occurrence + linear intensity",
         "linear occurrence + seasonal intensity"),
        ("GP occurrence replacing linear occurrence; intensity fixed linear",
         "GP occurrence + linear intensity",
         "linear occurrence + linear intensity"),
        ("GP intensity replacing linear intensity; occurrence fixed linear",
         "linear occurrence + GP intensity",
         "linear occurrence + linear intensity"),
        ("complete GP minus complete linear",
         "GP occurrence + GP intensity", "linear occurrence + linear intensity"),
    ]
    contrast_rows = []
    for label, new_name, old_name in comparisons:
        new, old = scores[new_name], scores[old_name]
        lo, hi = paired_interval(new, old, reference, block=14,
                                 seed=20261023)
        contrast_rows.append({"city": city, "contrast": label,
                              "gain": (old.mean()-new.mean())/reference.mean(),
                              "ci_low": lo, "ci_high": hi})
    pd.DataFrame(score_rows).to_csv(
        OUT/f"{city}_corrected_feature_factorial_scores.csv", index=False)
    result = pd.DataFrame(contrast_rows)
    result.to_csv(OUT/f"{city}_corrected_feature_factorial_contrasts.csv",
                  index=False)
    result["linear_lambda_occ"] = linear["lambda_occ"]
    result["linear_lambda_tail"] = linear["lambda_tail"]
    result["linear_xi"] = linear["xi"]
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name)
