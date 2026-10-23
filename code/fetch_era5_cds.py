"""
fetch_era5_cds.py — fetch the convective / moisture-transport ERA5 fields that
WeatherBench2 lacks (CAPE, IVT) from the Copernicus CDS, for the London box.

These are the untested lever for *intensity* predictability (CAPE relates to
convective rainfall intensity; IVT to moisture supply). CDS subsets the box
server-side, so the download stays small.

ONE-TIME SETUP (only you can do these):
  1. Put your CDS Personal Access Token in ~/.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <your token from https://cds.climate.copernicus.eu/profile>
  2. Accept the licence for "ERA5 hourly data on single levels from 1940 to
     present" at the bottom of its CDS download form (once).

RUN:
  python3 fetch_era5_cds.py --test                      # 1 month, checks auth
  python3 fetch_era5_cds.py --start 1989 --end 2018 \
          --out era5_cape_ivt_daily.parquet

Output: daily cape_mean / cape_max / ivt_mean / ivt_max over the box. Merge with
the WeatherBench2 features and re-run the bake-off.
"""
import argparse
import numpy as np
import pandas as pd

DATASET = "reanalysis-era5-single-levels"
# Physically-motivated, NON-CIRCULAR antecedent predictors WB2 lacks.
# Convective-instability + moisture diagnostics. NO precipitation fields
# (the target is itself ERA5-derived precip -> any precip field would be circular).
VARS = ["convective_available_potential_energy",      # CAPE
        "convective_inhibition",                       # CIN
        "k_index",                                     # K-index (instability)
        "total_totals_index",                          # Total Totals (instability)
        "total_column_water_vapour",                   # column moisture
        "vertical_integral_of_eastward_water_vapour_flux",   # IVT east
        "vertical_integral_of_northward_water_vapour_flux"]  # IVT north
AREA = [60, -20, 45, 10]                    # N, W, S, E  (London + upstream)
TIMES = ["00:00", "06:00", "12:00", "18:00"]  # 6-hourly is enough for daily stats
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def _retrieve_year(client, year, path, months=MONTHS):
    client.retrieve(DATASET, {
        "product_type": ["reanalysis"],
        "variable": VARS,
        "year": [str(year)],
        "month": months,
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, path)


def _daily_features(nc_path):
    """Generic: regional daily mean+max of every variable in the file, plus IVT
    magnitude derived from the eastward/northward water-vapour-flux components."""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    tname = "valid_time" if "valid_time" in ds.dims else "time"   # new CDS uses valid_time
    if tname != "time":
        ds = ds.rename({tname: "time"})
    dmean = ds.resample(time="1D").mean()
    dmax = ds.resample(time="1D").max()
    f = pd.DataFrame(index=dmean["time"].values)
    rm = lambda da: da.mean(dim=["latitude", "longitude"]).values
    east = north = None
    for v in ds.data_vars:
        f[f"{v}_mean"] = rm(dmean[v])
        f[f"{v}_max"] = rm(dmax[v])
        ln = str(ds[v].attrs.get("long_name", "")).lower()
        if "eastward" in ln and "water vapour" in ln:
            east = v
        if "northward" in ln and "water vapour" in ln:
            north = v
    if east and north:                                # IVT magnitude (moisture transport)
        f["ivt_mean"] = rm(np.sqrt(dmean[east] ** 2 + dmean[north] ** 2))
    f.index.name = "date"
    return f


def build(start, end, out, test=False):
    import cdsapi, os, tempfile
    client = cdsapi.Client()
    years = [start] if test else list(range(start, end + 1))
    parts = []
    for yr in years:
        tmp = os.path.join(tempfile.gettempdir(), f"era5cds_{yr}.nc")
        print(f"  retrieving {yr} ...", flush=True)
        _retrieve_year(client, yr, tmp, months=["01"] if test else MONTHS)
        f = _daily_features(tmp)
        if test:                              # keep just one month in test mode
            f = f.iloc[:31]
        parts.append(f)
        feats = pd.concat(parts).sort_index()
        feats.index.name = "date"
        feats.to_parquet(out) if out.endswith(".parquet") else feats.to_csv(out)
        print(f"  {yr}: {f.shape[0]} days, {f.shape[1]} feats  [saved {feats.shape[0]} rows]",
              flush=True)
        try:
            os.remove(tmp)
        except OSError:
            pass
    print(f"\nDONE: {feats.shape} -> {out}\n{feats.head()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1989)
    ap.add_argument("--end", type=int, default=2018)
    ap.add_argument("--out", default="era5_cape_ivt_daily.parquet")
    ap.add_argument("--area", default=None, help="N,W,S,E box (default London)")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.area:
        AREA = [float(x) for x in a.area.split(",")]
    build(a.start, a.end, a.out, test=a.test)
