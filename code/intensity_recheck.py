"""Resolve the inconsistent fine/coarse conditional-intensity results.

One model is selected with blocked time-series cross-validation.  Its predictions
are then scored on 19- and 9-level quantile grids, so a grid comparison cannot
silently change the fitted model or penalty.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit

from data import load_real, standardize
from major_revision_analysis import (OUT, U, city_paths, paired_gain_ci,
                                     stage_drivers)


def pinball(y, pred, q):
    e = y - pred
    return np.maximum(q*e, (q-1)*e)


def fit_predict_quantiles(xtr, ytr, xte, qs):
    cv = TimeSeriesSplit(n_splits=3)
    pred = np.empty((len(xte), len(qs)))
    chosen = []
    for j, q in enumerate(qs):
        best_alpha, best_loss = None, np.inf
        for alpha in (0.01, 0.03, 0.05, 0.1, 0.3):
            losses = []
            for ti, vi in cv.split(xtr):
                model = QuantileRegressor(quantile=q, alpha=alpha,
                                          solver="highs").fit(xtr[ti], ytr[ti])
                losses.append(pinball(ytr[vi],
                                      np.clip(model.predict(xtr[vi]), 0, None), q).mean())
            if np.mean(losses) < best_loss:
                best_alpha, best_loss = alpha, np.mean(losses)
        model = QuantileRegressor(quantile=q, alpha=best_alpha,
                                  solver="highs").fit(xtr, ytr)
        pred[:, j] = np.clip(model.predict(xte), 0, None)
        chosen.append(best_alpha)
    # Increasing rearrangement handles crossing without changing marginal fits.
    return np.sort(pred, axis=1), chosen


def per_case_crps(y, pred, clim, qs):
    weights = np.gradient(qs)
    loss_model = np.column_stack([pinball(y, pred[:, j], q)
                                  for j, q in enumerate(qs)])
    loss_clim = np.column_stack([pinball(y, np.full(len(y), clim[j]), q)
                                 for j, q in enumerate(qs)])
    return 2*(loss_model*weights).sum(1), 2*(loss_clim*weights).sum(1)


def run_city(city):
    drivers = stage_drivers(city)[-1][1]
    d, dates = load_real(city_paths(city)["index"], n_lags=14, u=U,
                         horizon=1, drivers_df=drivers)
    years = dates.year.to_numpy()
    rows = []
    qs_fine = np.arange(.05, .951, .05)
    qs_coarse = np.arange(.1, .901, .1)
    for cut_year in (2002, 2005, 2008, 2011):
        tr, te = years <= cut_year, years > cut_year
        xtr, xte = standardize(d["X"][tr], d["X"][te])
        xtr, xte = xtr.numpy(), xte.numpy()
        itr, ite = d["I"][tr].numpy(), d["I"][te].numpy()
        etr, ete = itr > U, ite > U
        ytr, yte = itr[etr]-U, ite[ete]-U
        pred, alphas = fit_predict_quantiles(xtr[etr], ytr, xte[ete], qs_fine)
        clim = np.quantile(ytr, qs_fine)
        for grid, idx in (("fine-19", np.arange(len(qs_fine))),
                          ("coarse-9", np.arange(1, len(qs_fine), 2))):
            cm, cc = per_case_crps(yte, pred[:, idx], clim[idx], qs_fine[idx])
            # paired_gain_ci(new, old, denominator) reduces here to skill and
            # uses the same fixed 2,000 moving-block resamples.
            sk, lo, hi = paired_gain_ci(cm, cc, cc,
                                        seed=20261015+cut_year+len(idx),
                                        block=10)
            rows.append(dict(city=city, test_period=f"{cut_year+1}-2018",
                             n_train_extreme=int(etr.sum()),
                             n_test_extreme=int(ete.sum()), grid=grid,
                             CRPS_skill=sk, ci_low=lo, ci_high=hi,
                             alpha_median=float(np.median(alphas))))
        print(pd.DataFrame(rows[-2:]).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(OUT/f"{city}_intensity_grid_recheck.csv", index=False)


if __name__ == "__main__":
    for CITY in ("london", "paris"):
        run_city(CITY)
