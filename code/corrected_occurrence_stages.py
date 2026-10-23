"""Brier-skill stages for the corrected predictor groups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from predictor_group_sensitivity import corrected_driver_groups
from second_review_analyses import (
    OCC_LAMBDAS, OUT, load_with_drivers, paired_interval, seasonal_parameters,
)


def fit_occurrence_cv(x_train, y_train, x_test):
    n = len(x_train)
    folds = [(0, int(.55*n), int(.70*n)),
             (0, int(.70*n), int(.85*n)),
             (0, int(.85*n), n)]
    losses = {lam: [] for lam in OCC_LAMBDAS}
    for _, train_end, valid_end in folds:
        tr = np.arange(train_end)
        va = np.arange(train_end, valid_end)
        mu = x_train[tr].mean(0)
        sd = x_train[tr].std(0) + 1e-8
        xtr = (x_train[tr]-mu)/sd
        xva = (x_train[va]-mu)/sd
        for lam in OCC_LAMBDAS:
            model = LogisticRegression(C=1/lam, solver="lbfgs", max_iter=3000)
            model.fit(xtr.numpy(), y_train[tr])
            p = np.clip(model.predict_proba(xva.numpy())[:, 1], 1e-10, 1-1e-10)
            y = y_train[va]
            losses[lam].append(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    lam = min(OCC_LAMBDAS, key=lambda v: np.mean(losses[v]))
    mu = x_train.mean(0)
    sd = x_train.std(0) + 1e-8
    model = LogisticRegression(C=1/lam, solver="lbfgs", max_iter=5000)
    model.fit(((x_train-mu)/sd).numpy(), y_train)
    return model.predict_proba(((x_test-mu)/sd).numpy())[:, 1], lam


def analyse(city):
    tele, moisture, convective = corrected_driver_groups(city)
    specifications = {
        "index history": None,
        "+ teleconnections": tele,
        "+ circulation/moisture": pd.concat([tele, moisture], axis=1),
        "+ convective instability": pd.concat([tele, moisture, convective], axis=1),
    }
    fitted = {}
    base_data = base_dates = base_cut = None
    for name, drivers in specifications.items():
        data, dates, cut = load_with_drivers(city, drivers)
        p, lam = fit_occurrence_cv(data["X"][:cut], data["O"][:cut].numpy(),
                                   data["X"][cut:])
        fitted[name] = (p, lam)
        base_data, base_dates, base_cut = data, dates, cut
    event = base_data["O"][base_cut:].numpy()
    seasonal = seasonal_parameters(base_data, base_dates, base_cut)
    ref = (seasonal["pi"]-event)**2
    rows = []
    previous = None
    for name in specifications:
        p, lam = fitted[name]
        score = (p-event)**2
        row = {"city": city, "stage": name, "BSS_vs_seasonal": 1-score.mean()/ref.mean(),
               "lambda_occ": lam, "increment": np.nan,
               "ci_low": np.nan, "ci_high": np.nan}
        if previous is not None:
            old = (fitted[previous][0]-event)**2
            lo, hi = paired_interval(score, old, ref, block=14, seed=20261023)
            row.update(increment=(old.mean()-score.mean())/ref.mean(),
                       ci_low=lo, ci_high=hi)
        rows.append(row)
        previous = name
    result = pd.DataFrame(rows)
    result.to_csv(OUT/f"{city}_corrected_occurrence_stages.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name)
