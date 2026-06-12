"""
conformal.py — tail-targeted, time-series conformal calibration.

Implements:
* split_conformal_upper : one-sided upper-bound calibration (the extreme tail),
                          with optional GPD tail-extrapolation of the score
                          quantile when the level is too high for the
                          calibration set to resolve (cf. Pasche et al. 2025).
* adaptive_conformal    : online ACI update (Gibbs & Candes 2021) giving
                          long-run coverage under temporal dependence.

Scores are I - qhi(x): the signed amount by which the observation exceeds the
model's upper quantile.  Calibrating their (1-alpha) quantile and adding it to
qhi yields a conformal upper bound with the target coverage.
"""
import numpy as np
from scipy.stats import genpareto


def _evt_extrapolate(scores, level):
    """GPD tail-extrapolation of an extreme quantile of `scores`.
    Used when level is so high that the empirical quantile saturates."""
    s = np.sort(scores)
    k = max(20, int(0.2 * len(s)))                 # top-k for the tail fit
    thr = s[-k]
    excess = s[-k:] - thr
    try:
        xi, _, beta = genpareto.fit(excess, floc=0.0)
    except Exception:
        return s[-1]
    m, n = k, len(s)
    # quantile: thr + beta/xi [ ((n/m)(1-level))^{-xi} - 1 ]
    r = (n / m) * (1 - level)
    if abs(xi) < 1e-4:
        return thr - beta * np.log(max(r, 1e-12))
    return thr + beta / xi * (max(r, 1e-12) ** (-xi) - 1)


def split_conformal_upper(y_cal, qhi_cal, y_test, qhi_test, alpha=0.1):
    """Conformalise an upper bound to target 1-alpha coverage.
    Returns (upper_bounds_test, coverage, q_hat)."""
    scores = np.asarray(y_cal) - np.asarray(qhi_cal)
    n = len(scores)
    level = (1 - alpha) * (1 + 1.0 / n)
    if level >= 1.0:                                # too few calib points -> extrapolate
        q_hat = _evt_extrapolate(scores, 1 - alpha)
    else:
        q_hat = np.quantile(scores, level, method="higher")
    upper = np.asarray(qhi_test) + q_hat
    coverage = float(np.mean(np.asarray(y_test) <= upper))
    return upper, coverage, float(q_hat)


def adaptive_conformal(y_seq, qhi_seq, scores_init, alpha=0.1, gamma=0.02):
    """Online ACI over a test sequence; returns (upper_seq, running_coverage).
    alpha_t is nudged up/down to hold long-run coverage at 1-alpha."""
    y_seq, qhi_seq = np.asarray(y_seq), np.asarray(qhi_seq)
    pool = list(np.asarray(scores_init))
    a_t, covered = alpha, []
    upper = np.empty(len(y_seq))
    for t in range(len(y_seq)):
        lvl = min(max(1 - a_t, 0.0), 1.0)
        q_hat = (np.quantile(pool, lvl, method="higher")
                 if 0 < lvl < 1 else (max(pool) if lvl >= 1 else min(pool)))
        upper[t] = qhi_seq[t] + q_hat
        err = int(y_seq[t] > upper[t])              # miscoverage
        covered.append(1 - err)
        a_t = a_t + gamma * (alpha - err)           # ACI update (eq. 18)
        pool.append(y_seq[t] - qhi_seq[t])          # grow calibration pool
    return upper, float(np.mean(covered))
