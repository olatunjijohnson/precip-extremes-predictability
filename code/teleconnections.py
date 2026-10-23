"""
teleconnections.py — download & parse large-scale climate indices (NAO, AO,
ENSO) as antecedent predictors.  These are low-frequency monthly series; we
broadcast them to daily and the model lags them (no leakage).

Sources (machine-readable, verified formats):
  * NAO     : NOAA PSL  https://psl.noaa.gov/data/correlation/nao.data      (PSL .data)
  * Nino3.4 : NOAA PSL  https://psl.noaa.gov/data/correlation/nina34.data   (PSL .data)
  * AO      : NOAA CPC  monthly.ao.index.b50.current.ascii                  (year month value)
  * ENSO ONI: NOAA CPC  oni.ascii.txt                                       (SEAS YR TOTAL ANOM)

If a URL drifts, edit INDICES below — the parsers are robust to extra header /
footer lines.  Run standalone to cache a CSV:  python3 teleconnections.py
"""
import io, os, urllib.request
import numpy as np
import pandas as pd

SEASON_TO_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                   "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def parse_psl(text):
    """PSL .data: 'year v1..v12', sentinel -99.90.  -> {(year,month): value}."""
    rec = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 13:
            try:
                yr, vals = int(p[0]), [float(x) for x in p[1:]]
            except ValueError:
                continue
            for m, v in enumerate(vals, 1):
                if v > -99.0:                       # skip -99.90 sentinel
                    rec[(yr, m)] = v
    return rec


def parse_ym_value(text):
    """'year month value' (e.g. CPC monthly AO).  -> {(year,month): value}."""
    rec = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= 3:
            try:
                yr, mo, v = int(p[0]), int(p[1]), float(p[2])
            except ValueError:
                continue
            if 1 <= mo <= 12:
                rec[(yr, mo)] = v
    return rec


def parse_oni(text):
    """CPC ONI: 'SEAS YR TOTAL ANOM'.  -> {(year,month): anomaly}."""
    rec = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 4 and p[0] in SEASON_TO_MONTH:
            try:
                rec[(int(p[1]), SEASON_TO_MONTH[p[0]])] = float(p[3])
            except ValueError:
                continue
    return rec


INDICES = {
    "nao":    ("https://psl.noaa.gov/data/correlation/nao.data", parse_psl),
    "nino34": ("https://psl.noaa.gov/data/correlation/nina34.data", parse_psl),
    "ao":     ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
               "daily_ao_index/monthly.ao.index.b50.current.ascii", parse_ym_value),
    # alternative ENSO index (uncomment to use instead of/with nino34):
    # "oni":  ("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", parse_oni),
}


def _rec_to_series(rec):
    idx = [pd.Timestamp(year=y, month=m, day=1) for (y, m) in rec]
    return pd.Series(list(rec.values()), index=pd.DatetimeIndex(idx)).sort_index()


def fetch_teleconnections(start="1989-01-01", end="2018-12-31",
                          cache_dir=None, indices=None):
    """Return a DAILY DataFrame of the indices over [start,end] (monthly values
    forward-filled to daily).  Robust: skips any index that fails to download."""
    indices = indices or INDICES
    cols = {}
    for name, (url, parser) in indices.items():
        try:
            raw = _get(url, cache_dir, name)
            s = _rec_to_series(parser(raw))
            if len(s) == 0:
                raise ValueError("parsed 0 records")
            cols[name] = s
            print(f"  [{name}] {len(s)} monthly values "
                  f"{s.index.min().date()}..{s.index.max().date()}")
        except Exception as e:
            print(f"  [{name}] SKIPPED ({type(e).__name__}: {e})")
    if not cols:
        raise RuntimeError("no teleconnection indices could be loaded")
    monthly = pd.DataFrame(cols).sort_index()
    daily = (monthly.resample("D").ffill()
             .reindex(pd.date_range(start, end, freq="D")).ffill().bfill())
    daily.index.name = "date"
    return daily


def _get(url, cache_dir, name):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fp = os.path.join(cache_dir, f"{name}.txt")
        if os.path.exists(fp):
            return open(fp).read()
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode("utf-8", "ignore")
    if cache_dir:
        open(os.path.join(cache_dir, f"{name}.txt"), "w").write(raw)
    return raw


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    df = fetch_teleconnections(cache_dir=os.path.join(here, "_telecon_cache"))
    out = os.path.join(here, "teleconnections_daily.csv")
    df.to_csv(out)
    print(f"\nsaved {df.shape} -> {out}")
    print(df.tail())
