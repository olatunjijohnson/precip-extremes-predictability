"""
evaluation.py — calibration-aware, proper-scoring evaluation.

The decisive question: is the model calibrated AND sharper than climatology, and
does it beat simple baselines on a tail-focused proper score?

Metrics
-------
* brier_decomposition : Brier score + Murphy's reliability/resolution/uncertainty
                        decomposition + Brier skill score vs climatology.
                        RES > 0 with low REL = calibrated AND informative.
* twcrps_*            : threshold-weighted CRPS (Gneiting & Ranjan 2011) with
                        weight 1{z > tau}.  With tau = u the bulk distribution is
                        irrelevant, so the model predictive needs only (pi,sigma,xi).
* reliability_curve  : calibration-diagram data.

A positive Brier/twCRPS skill score over climatology is the bar the model must
clear to justify its complexity.
"""
import numpy as np
from scipy.stats import genpareto


# ---------------------------------------------------------------- occurrence
def brier_decomposition(y, p, n_bins=10):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 0, 1)
    obar = y.mean()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    REL = RES = 0.0
    N = len(y)
    for k in range(n_bins):
        m = idx == k
        nk = m.sum()
        if nk == 0:
            continue
        pk, ok = p[m].mean(), y[m].mean()
        REL += nk * (pk - ok) ** 2
        RES += nk * (ok - obar) ** 2
    REL /= N; RES /= N
    UNC = obar * (1 - obar)
    BS = np.mean((p - y) ** 2)
    BSS = 1 - BS / UNC if UNC > 0 else float("nan")
    return dict(BS=BS, REL=REL, RES=RES, UNC=UNC, BSS=BSS)


def reliability_curve(y, p, n_bins=10):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for k in range(n_bins):
        m = idx == k
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


# ---------------------------------------------------------------- twCRPS
def _gpd_cdf(yexc, sigma, xi):
    yexc = np.clip(yexc, 0, None)
    if abs(xi) < 1e-6:
        return 1 - np.exp(-yexc / sigma)
    base = np.clip(1 + xi * yexc / sigma, 1e-9, None)
    return 1 - base ** (-1.0 / xi)


def _grid(tau, zmax, nz):
    z = np.linspace(tau, zmax, nz)
    return z, (z[1] - z[0])


def make_grid(tau=1.0, zmax=8.0, nz=400):
    return _grid(tau, zmax, nz)


def twcrps_from_cdf(y, F_grid, z, dz):
    """Generic twCRPS from a precomputed predictive CDF on the grid.
    y:(n,)  F_grid:(n,nz)  z:(nz,)  -> per-point twCRPS (n,)."""
    y = np.asarray(y, float)
    ind = (z[None, :] >= y[:, None]).astype(float)
    return ((np.asarray(F_grid) - ind) ** 2).sum(1) * dz


def twcrps_model(y, pi, sigma, xi, u=1.0, tau=1.0, zmax=8.0, nz=400):
    """Per-point twCRPS for the hurdle+GPD predictive (tau >= u)."""
    y = np.asarray(y, float); pi = np.asarray(pi, float); sigma = np.asarray(sigma, float)
    z, dz = _grid(tau, zmax, nz)
    H = _gpd_cdf(z[None, :] - u, sigma[:, None], xi)            # (n,nz)
    F = (1 - pi[:, None]) + pi[:, None] * H
    ind = (z[None, :] >= y[:, None]).astype(float)
    return ((F - ind) ** 2).sum(1) * dz


def twcrps_climatology(y, train_index, tau=1.0, zmax=8.0, nz=400):
    y = np.asarray(y, float); tr = np.asarray(train_index, float)
    z, dz = _grid(tau, zmax, nz)
    Fc = (tr[:, None] <= z[None, :]).mean(0)                    # empirical CDF (nz,)
    ind = (z[None, :] >= y[:, None]).astype(float)
    return ((Fc[None, :] - ind) ** 2).sum(1) * dz


def fit_pot(train_index, u=1.0):
    """Stationary peaks-over-threshold baseline: constant exceedance prob + GPD."""
    tr = np.asarray(train_index, float)
    pi = float((tr > u).mean())
    exc = tr[tr > u] - u
    xi, _, sigma = genpareto.fit(exc, floc=0.0)
    return pi, float(sigma), float(xi)


def twcrps_pot(y, pi, sigma, xi, u=1.0, tau=1.0, zmax=8.0, nz=400):
    n = len(y)
    return twcrps_model(y, np.full(n, pi), np.full(n, sigma), xi, u, tau, zmax, nz)


def skill(score_model, score_ref):
    """Skill score: 1 - mean(model)/mean(ref).  >0 = better than reference."""
    return 1 - np.mean(score_model) / np.mean(score_ref)
