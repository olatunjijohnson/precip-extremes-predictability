"""Check whether GP forecasts are scored as plug-ins or predictive mixtures.

The existing paper averages scores across seed-specific plug-in forecasts.  This
script compares that calculation with mixtures of seed forecasts and stored
latent posterior draws.  It writes diagnostics only; it does not edit the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from second_review_analyses import exact_twcrps_parts


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def survival(y, sigma, xi):
    base = 1.0 + xi * y / sigma
    if abs(xi) < 1e-7:
        return np.exp(-y/sigma)
    return np.where(base > 0, np.maximum(base, 0.0) ** (-1.0/xi), 0.0)


def mixture_score(index, pi_components, sigma_components, xi_components,
                  ymax=50.0, chunk=24):
    """Numerically integrate the score of F with survivor E[pi*S]."""
    index = np.asarray(index, float)
    pi_components = np.asarray(pi_components, float)
    sigma_components = np.asarray(sigma_components, float)
    xi_components = np.asarray(xi_components, float)
    # Dense close to the threshold and logarithmic in the remote tail.
    grid = np.unique(np.r_[np.linspace(0, 2, 401),
                           np.geomspace(2.01, ymax, 400)])
    scores = np.empty(len(index))
    rewards = np.empty(len(index))
    penalties = np.empty(len(index))
    observations = np.maximum(index-1.0, 0.0)
    events = index > 1.0
    for start in range(0, len(index), chunk):
        stop = min(start+chunk, len(index))
        sig = sigma_components[:, start:stop, None]
        pi = pi_components[:, start:stop, None]
        xi = xi_components[:, None, None]
        y = grid[None, None, :]
        base = 1.0 + xi*y/sig
        surv_exp = np.exp(-y/sig)
        safe_xi = np.where(np.abs(xi) < 1e-7, 1e-7, xi)
        surv_gpd = np.where(base > 0,
                            np.maximum(base, 1e-300) ** (-1.0/safe_xi), 0.0)
        surv = np.where(np.abs(xi) < 1e-7, surv_exp, surv_gpd)
        q = np.mean(pi*surv, axis=0)
        penalties[start:stop] = np.trapezoid(q*q, grid, axis=1)
        for local, row in enumerate(range(start, stop)):
            if events[row]:
                yy = observations[row]
                mask = grid < yy
                x = np.r_[grid[mask], yy]
                if len(x) > 1:
                    # Interpolate q to the observation when it is off-grid.
                    qy = np.interp(x, grid, q[local])
                    rewards[row] = -2*np.trapezoid(qy, x)
                else:
                    rewards[row] = 0.0
            else:
                rewards[row] = 0.0
        scores[start:stop] = (observations[start:stop]
                              + rewards[start:stop] + penalties[start:stop])
    return scores, observations, rewards, penalties


def analyse(city, corrected=False):
    prefix = "corrected_" if corrected else ""
    d = np.load(OUT/f"{city}_{prefix}gp_main_predictions.npz")
    index = d["observed_index"]
    pi, sigma, xi = d["pi"], d["sigma"], d["xi"]
    plugin_scores = np.mean([
        exact_twcrps_parts(index, pi[s], sigma[s], float(xi[s]))[0]
        for s in range(len(xi))], axis=0)

    seed_mix, *_ = mixture_score(index, pi, sigma, xi)

    # Flatten seed and within-seed posterior draws into one predictive mixture.
    pi_draw = d["pi_draw"].reshape(-1, len(index))
    sigma_draw = d["sigma_draw"].reshape(-1, len(index))
    xi_draw = np.repeat(xi, d["pi_draw"].shape[1])
    posterior_mix, obs, reward, penalty = mixture_score(
        index, pi_draw, sigma_draw, xi_draw)

    rows = [
        dict(city=city, calculation="mean of seed plug-in scores",
             mean_twCRPS=plugin_scores.mean()),
        dict(city=city, calculation="mixture of seed plug-in CDFs",
             mean_twCRPS=seed_mix.mean()),
        dict(city=city, calculation="mixture over posterior draws and seeds",
             mean_twCRPS=posterior_mix.mean()),
    ]
    pd.DataFrame(rows).to_csv(OUT/f"{city}_{prefix}gp_mixture_check.csv", index=False)
    pd.DataFrame([dict(city=city, observation_term=obs.mean(),
                       event_reward=reward.mean(), penalty=penalty.mean(),
                       total=posterior_mix.mean())]).to_csv(
        OUT/f"{city}_{prefix}gp_mixture_decomposition.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    for city_name in ("london", "paris"):
        analyse(city_name, corrected=True)
