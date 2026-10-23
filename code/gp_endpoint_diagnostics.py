"""Check finite endpoints implied by negative GP-EVT shape estimates."""
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
rows = []
for feature_set, suffix in (("original 49", "gp_main_predictions.npz"),
                            ("corrected 48", "corrected_gp_main_predictions.npz")):
  for city in ("london", "paris"):
    z = np.load(OUT/f"{city}_{suffix}")
    event = z["observed_event"].astype(bool)
    excess = z["observed_index"][event] - 1.0
    for seed, xi in enumerate(z["xi"]):
        if xi < 0:
            endpoint = z["sigma"][seed, event] / abs(float(xi))
            ratio = excess / endpoint
            rows.append({
                "feature_set": feature_set, "city": city, "seed": seed, "xi": float(xi),
                "n_test_extremes": int(event.sum()),
                "n_at_or_beyond_endpoint": int((ratio >= 1).sum()),
                "maximum_excess_to_endpoint_ratio": float(ratio.max()),
                "median_excess_to_endpoint_ratio": float(np.median(ratio)),
                "minimum_endpoint_minus_excess": float((endpoint-excess).min()),
            })
        else:
            rows.append({
                "feature_set": feature_set, "city": city, "seed": seed, "xi": float(xi),
                "n_test_extremes": int(event.sum()),
                "n_at_or_beyond_endpoint": 0,
                "maximum_excess_to_endpoint_ratio": np.nan,
                "median_excess_to_endpoint_ratio": np.nan,
                "minimum_endpoint_minus_excess": np.nan,
            })
result = pd.DataFrame(rows)
result.to_csv(OUT/"gp_finite_endpoint_diagnostics.csv", index=False)
print(result.to_string(index=False))
