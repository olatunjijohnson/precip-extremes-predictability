"""Gaussian-process sensitivity analyses.

It writes fitted predictions and diagnostics to ``results``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluation as ev
from data import load_drivers, load_real, standardize
from major_revision_analysis import (city_paths, seasonal_event_prob,
                                     fit_seasonal_gpd, seasonal_pot_scores)
from tcdgp import TCDGP


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
U = 1.0
MAIN_SEEDS = (0, 1, 2, 3, 4)
SENS_SEEDS = (0, 1, 2)


class LinearBiasKernel(nn.Module):
    """Linear covariance plus a constant covariance for a latent intercept."""

    def __init__(self, d: int):
        super().__init__()
        self.d = float(d)
        self.log_os = nn.Parameter(torch.zeros(()))
        self.log_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x1, x2):
        return (torch.exp(self.log_os) * (x1 @ x2.t()) / self.d
                + torch.exp(self.log_bias))

    def diag(self, x):
        return (torch.exp(self.log_os) * (x * x).sum(1) / self.d
                + torch.exp(self.log_bias))


def load_city(city: str):
    paths = city_paths(city)
    tele = load_drivers(paths["tele"], lag_monthly_teleconnections=True)
    era = load_drivers(paths["era"], extra_era5_lag_days=1)
    conv = load_drivers(paths["conv"], extra_era5_lag_days=1)
    drivers = pd.concat([tele, era, conv], axis=1)
    drivers = drivers.loc[:, ~drivers.columns.duplicated()]
    data, dates = load_real(paths["index"], n_lags=14, u=U, horizon=1,
                            drivers_df=drivers)
    cut = int((dates <= np.datetime64("2008-12-31")).sum())
    xtr, xte = standardize(data["X"][:cut], data["X"][cut:])
    return data, dates, cut, xtr, xte


def references(data, dates, cut):
    itr = data["I"][:cut].numpy()
    ite = data["I"][cut:].numpy()
    otr = data["O"][:cut].numpy()
    ote = data["O"][cut:].numpy()
    train_dates, test_dates = dates[:cut], dates[cut:]
    p_seas = seasonal_event_prob(otr, train_dates, test_dates)
    bs_seas = (p_seas - ote) ** 2
    theta, _ = fit_seasonal_gpd(itr, train_dates)
    zmax = float(max(itr.max(), ite.max()) * 1.05)
    tw_seas, _ = seasonal_pot_scores(ite, test_dates, p_seas, theta, zmax)
    return ite, ote, bs_seas, tw_seas, zmax


def fit_one(xtr, xte, data, cut, *, seed, M=64, coupled=True,
            kernel="rbf", n_draws=40):
    torch.manual_seed(seed)
    model = TCDGP(d_in=xtr.shape[1], u=U, M=M, Q=2,
                  coupled=coupled, deep=False)
    if kernel == "linear_bias":
        model.gp.kernel = LinearBiasKernel(xtr.shape[1])
    model.fit(xtr, data["O"][:cut], data["excess"][:cut], epochs=600,
              lr=0.02, batch=1024, verbose=False)

    # Match the reported plug-in calculation: posterior means of pi and sigma.
    torch.manual_seed(10000 + seed)
    params = model.predict_params(xte, n_samples=200)
    pi = params["pi"].numpy().astype("float32")
    sigma = params["sigma"].numpy().astype("float32")
    xi = float(params["xi"])

    # Preserve latent draws so that a posterior-mixture score can be checked.
    torch.manual_seed(20000 + seed)
    with torch.no_grad():
        latent = model.gp.sample_latent(model.fe(xte), n_draws)
        pi_draw = torch.sigmoid(latent[..., 0]).cpu().numpy().astype("float32")
        sigma_draw = (F.softplus(latent[..., 1]) + 1e-4).cpu().numpy().astype("float32")
    return dict(pi=pi, sigma=sigma, xi=xi, pi_draw=pi_draw,
                sigma_draw=sigma_draw, rho=model.gp.rho(),
                lengthscales=np.exp(model.gp.kernel.log_ls.detach().cpu().numpy())
                if hasattr(model.gp.kernel, "log_ls") else np.array([]))


def analyse_city(city: str):
    data, dates, cut, xtr, xte = load_city(city)
    ite, ote, bs_ref, tw_ref, zmax = references(data, dates, cut)
    configurations = [
        ("rbf_coupled_M64", 64, True, "rbf", MAIN_SEEDS, True),
        ("rbf_independent_M64", 64, False, "rbf", SENS_SEEDS, False),
        ("rbf_coupled_M32", 32, True, "rbf", SENS_SEEDS, False),
        ("rbf_coupled_M128", 128, True, "rbf", SENS_SEEDS, False),
        ("linear_coupled_M64", 64, True, "linear_bias", SENS_SEEDS, False),
    ]
    rows = []
    for name, M, coupled, kernel, seeds, save_draws in configurations:
        print(f"\n{city}: {name}", flush=True)
        fitted = []
        for seed in seeds:
            result = fit_one(xtr, xte, data, cut, seed=seed, M=M,
                             coupled=coupled, kernel=kernel)
            bs = (result["pi"] - ote) ** 2
            tw = ev.twcrps_model(ite, result["pi"], result["sigma"],
                                 result["xi"], u=U, tau=U, zmax=zmax, nz=1200)
            row = dict(city=city, configuration=name, seed=seed, M=M,
                       coupled=coupled, kernel=kernel,
                       BSS=1-bs.mean()/bs_ref.mean(),
                       twCRPSS=1-tw.mean()/tw_ref.mean(),
                       mean_BS=bs.mean(), mean_twCRPS=tw.mean(),
                       xi=result["xi"], rho=result["rho"])
            rows.append(row)
            fitted.append(result)
            print(row, flush=True)
        if save_draws:
            np.savez_compressed(
                OUT / f"{city}_gp_main_predictions.npz",
                pi=np.stack([r["pi"] for r in fitted]),
                sigma=np.stack([r["sigma"] for r in fitted]),
                xi=np.array([r["xi"] for r in fitted]),
                pi_draw=np.stack([r["pi_draw"] for r in fitted]),
                sigma_draw=np.stack([r["sigma_draw"] for r in fitted]),
                rho=np.array([r["rho"] for r in fitted]),
                lengthscales=np.stack([r["lengthscales"] for r in fitted]),
                dates=dates[cut:].to_numpy(dtype="datetime64[D]"),
                observed_index=ite,
                observed_event=ote,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{city}_gp_sensitivity.csv", index=False)
    summary = (frame.groupby("configuration")
               .agg(BSS_mean=("BSS", "mean"), BSS_sd=("BSS", "std"),
                    twCRPSS_mean=("twCRPSS", "mean"),
                    twCRPSS_sd=("twCRPSS", "std"),
                    xi_mean=("xi", "mean"), xi_sd=("xi", "std"),
                    rho_mean=("rho", "mean"), rho_sd=("rho", "std"))
               .reset_index())
    summary.to_csv(OUT / f"{city}_gp_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    metadata = {"main_seeds": MAIN_SEEDS, "sensitivity_seeds": SENS_SEEDS,
                "epochs": 600, "batch": 1024, "learning_rate": 0.02,
                "posterior_draws_saved_per_seed": 40}
    (OUT / "gp_analysis_settings.json").write_text(json.dumps(metadata, indent=2))
    for city_name in ("london", "paris"):
        analyse_city(city_name)
