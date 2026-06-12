"""
fetch_city_index.py — download a second city's standardised precipitation-extreme
index from the same Copernicus product as London and process it to the identical
CSV format (index_max = daily maximum over the city's grid cells).

Validated: this reconstruction reproduces the supplied London index exactly.

Run (needs ~/.cdsapirc + accepted licence for
 sis-european-risk-extreme-precipitation-indicators):
  python3 fetch_city_index.py --city paris --pct 95
"""
import argparse, glob, os, tempfile, zipfile
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

DS = "sis-european-risk-extreme-precipitation-indicators"


def fetch(city, pct, y0=1989, y1=2018, out=None):
    out = out or f"../../data/{city}_precip_extreme_index_{pct}th_1989_2018.csv"
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), out))
    years = [str(y) for y in range(y0, y1 + 1)]
    dl = os.path.join(tempfile.gettempdir(), f"{city}_{pct}.download")
    print(f"retrieving {city} {pct}th, {y0}-{y1} ...", flush=True)
    cdsapi.Client().retrieve(DS, {
        "spatial_coverage": "city",
        "variable": "standardised_precipitation_exceeding_fixed_percentiles",
        "city": city,
        "product_type": "era5_2km",
        "temporal_aggregation": "daily",
        "percentile": f"{pct}th",
        "period": years,
    }, dl)

    # the download is a zip of yearly netCDFs (or a single nc)
    ncdir = os.path.join(tempfile.gettempdir(), f"{city}_{pct}_nc")
    os.makedirs(ncdir, exist_ok=True)
    if zipfile.is_zipfile(dl):
        with zipfile.ZipFile(dl) as z:
            z.extractall(ncdir)
        ncs = sorted(glob.glob(os.path.join(ncdir, "*.nc")))
    else:
        ncs = [dl]
    print(f"  {len(ncs)} netCDF files", flush=True)

    var = f"nrr{pct}p"
    frames = []
    for nc in ncs:
        ds = xr.open_dataset(nc)
        if var not in ds:                       # fall back to the single data var
            var = list(ds.data_vars)[0]
        da = ds[var]
        mx = da.max(dim=["rlat", "rlon"], skipna=True).to_series()
        mn = da.mean(dim=["rlat", "rlon"], skipna=True).to_series()
        n = int(da.isel(time=0).notnull().sum())
        frames.append(pd.DataFrame(
            {"index_max": mx.values, "index_mean": mn.values, "n_cells": n},
            index=pd.to_datetime(mx.index)))
    df = pd.concat(frames).sort_index()
    df.index.name = "date"
    df.to_csv(out)
    ext = int((df["index_max"] > 1).sum())
    print(f"\nDONE: {df.shape} -> {out}\n  extreme days (I>1): {ext} "
          f"({100*ext/len(df):.2f}%)\n{df.head(3)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="paris")
    ap.add_argument("--pct", type=int, default=95)
    ap.add_argument("--y0", type=int, default=1989)
    ap.add_argument("--y1", type=int, default=2018)
    a = ap.parse_args()
    fetch(a.city, a.pct, a.y0, a.y1)
