"""
data.py — synthetic generator (with KNOWN occurrence-intensity correlation,
for validating the dependent hurdle) and a real-data loader/feature builder.
"""
import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------
def make_synthetic(n=4000, d=5, rho=0.7, xi=0.2, base_rate=0.08, seed=0):
    """Generate a hurdle series whose occurrence and intensity latents have a
    KNOWN correlation `rho`, so the model's recovered rho can be checked.

    g_pi  = sqrt(rho) * c(X) + sqrt(1-rho) * a(X)
    g_sig = sqrt(rho) * c(X) + sqrt(1-rho) * b(X)
    => corr(g_pi, g_sig) = rho  (a, b, c independent linear maps of X).
    """
    rng = np.random.default_rng(seed)
    # smooth-ish covariates (light AR(1) so it resembles a time series)
    X = rng.standard_normal((n, d))
    X = np.cumsum(0.15 * X, axis=0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    wa, wb, wc = (rng.standard_normal((d,)) for _ in range(3))
    a, b, c = X @ wa, X @ wb, X @ wc
    norm = lambda z: (z - z.mean()) / (z.std() + 1e-8)
    a, b, c = norm(a), norm(b), norm(c)
    g_pi = np.sqrt(rho) * c + np.sqrt(1 - rho) * a
    g_sig = np.sqrt(rho) * c + np.sqrt(1 - rho) * b

    # occurrence: bias chosen to hit the target base rate
    from scipy.optimize import brentq
    rate = lambda bias: (1 / (1 + np.exp(-(1.3 * g_pi + bias)))).mean()
    bias = brentq(lambda β: rate(β) - base_rate, -20, 20)
    pi = 1 / (1 + np.exp(-(1.3 * g_pi + bias)))
    O = (rng.uniform(size=n) < pi).astype(float)

    # intensity: GPD excess on extreme days, scale driven by g_sig
    sigma = np.log1p(np.exp(0.6 * g_sig - 0.3)) + 0.05
    u = 1.0
    from scipy.stats import genpareto
    excess = genpareto.rvs(xi, scale=sigma, random_state=rng)
    excess = np.clip(excess, 0, None)
    I = np.where(O == 1, u + excess, rng.uniform(0, u, size=n))
    return _pack(X, I, u), {"rho": rho, "xi": xi, "rate": float(O.mean())}


# --------------------------------------------------------------------------
def load_real(csv_path, n_lags=14, u=1.0, horizon=1, drivers_df=None):
    """Load the London index CSV and build features to forecast the index
    `horizon` days ahead.

    For target day t = idx[i], the forecast origin is o = i - horizon: ALL
    predictors use information up to and including day o only (strictly
    antecedent — no leakage). Target seasonality (day-of-year of t) is known in
    advance and is allowed. If `drivers_df` is given, its columns are appended,
    aligned to the origin (shifted by `horizon`)."""
    df = (pd.read_csv(csv_path, parse_dates=["date"])
            .set_index("date").sort_index())
    df = df[["index_max"]].rename(columns={"index_max": "I"})
    s = df["I"]
    rows, dates = [], []
    idx = s.index
    start = n_lags + horizon - 1
    for i in range(start, len(s)):
        o = i - horizon                                  # forecast origin
        lags = s.iloc[o - n_lags + 1:o + 1].values[::-1]  # n_lags days ending at o
        hist = s.iloc[:o + 1]                            # up to and incl. origin
        doy = idx[i].day_of_year                         # target seasonality (known)
        feats = list(lags) + [
            hist.iloc[-7:].mean(), hist.iloc[-30:].mean(),
            hist.iloc[-7:].max(), (hist.iloc[-30:] > u).sum(),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ]
        rows.append(feats)
        dates.append(idx[i])
    X = np.asarray(rows)
    dates = pd.DatetimeIndex(dates)
    I = s.iloc[start:].values

    if drivers_df is not None:
        full = pd.date_range(idx.min(), idx.max(), freq="D")
        # forward-fill to daily, then shift to the origin => antecedent only
        dd = drivers_df.reindex(full).ffill().shift(horizon)
        drv = np.nan_to_num(dd.reindex(dates).to_numpy(dtype=float), nan=0.0)
        X = np.hstack([X, drv])
    return _pack(X, I, u), dates


def load_drivers(*paths):
    """Combine any number of daily driver tables (CSV/parquet, date-indexed) into
    one DataFrame for `load_real(..., drivers_df=...)`. Ignores None/missing.
    Returns None if nothing loadable."""
    frames = []
    for p in paths:
        if not p:
            continue
        d = (pd.read_parquet(p) if str(p).endswith(".parquet")
             else pd.read_csv(p, index_col=0, parse_dates=[0]))
        d.index = pd.DatetimeIndex(d.index)
        frames.append(d)
    if not frames:
        return None
    # align on dates; drop duplicate columns if a field appears in two files
    df = pd.concat(frames, axis=1).sort_index()
    return df.loc[:, ~df.columns.duplicated()]


# --------------------------------------------------------------------------
def _pack(X, I, u):
    O = (I > u).astype(float)
    excess = np.where(O == 1, I - u, 0.0)
    return {
        "X": torch.tensor(X, dtype=torch.float32),
        "I": torch.tensor(I, dtype=torch.float32),
        "O": torch.tensor(O, dtype=torch.float32),
        "excess": torch.tensor(excess, dtype=torch.float32),
        "u": u,
    }


def standardize(train_X, *other_X):
    mu, sd = train_X.mean(0), train_X.std(0) + 1e-8
    out = [(train_X - mu) / sd] + [(x - mu) / sd for x in other_X]
    return out if other_X else out[0]
