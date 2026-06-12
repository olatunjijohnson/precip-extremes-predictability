"""
fetch_era5_arco.py — pull antecedent ERA5 atmospheric drivers for London from
public cloud zarr and reduce them to a daily feature table.  No Copernicus
account / queue.

Two sources (default = WeatherBench2, much faster for regional extraction):
  * wb2  : WeatherBench2 ERA5 regridded to 240x121 (~1.5 deg), 6-hourly, 1959-2021.
           Small global footprint -> fast.  Has circulation + moisture fields,
           but NOT CAPE or IVT (use the CDS route for those; see ../02_era5_data.md).
  * arco : full-res ARCO-ERA5 (0.25 deg, hourly).  Has CAPE + IVT, but its global
           chunking makes regional bulk extraction very slow.

Requires:  pip install xarray zarr gcsfs dask scikit-learn pandas numpy pyarrow
Run where you have internet.

Examples:
  python3 fetch_era5_arco.py --list-vars
  python3 fetch_era5_arco.py --test
  python3 fetch_era5_arco.py --start 1989-01-01 --end 2018-12-31 \
                             --out era5_london_daily.parquet
"""
import argparse
import numpy as np
import pandas as pd

SOURCES = {
    "wb2": "gs://weatherbench2/datasets/era5/"
           "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    "arco": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
}

# Single-level fields common to both sources.
SINGLE_VARS = ["mean_sea_level_pressure", "total_column_water_vapour",
               "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
# ARCO-only extras (skipped automatically if absent).
ARCO_EXTRA = ["convective_available_potential_energy",
              "vertical_integral_of_eastward_water_vapour_flux",
              "vertical_integral_of_northward_water_vapour_flux"]
PLEVELS = {"geopotential": [500, 850], "specific_humidity": [850]}

# London-centred synoptic box (deg).  Robust to lon 0-360 and lat orientation.
LAT_S, LAT_N = 45, 60
LON_W, LON_E = -20, 10


def _open(source):
    import xarray as xr
    # dask chunking along time is essential for fast cloud reads
    return xr.open_zarr(SOURCES[source], storage_options={"token": "anon"},
                        chunks={"time": 256})


def _box(ds):
    """Select the London box by index, robust to coord orientation / 0-360 lon."""
    lat = ds.latitude.values
    lon = ds.longitude.values
    latm = (lat >= LAT_S) & (lat <= LAT_N)
    w, e = LON_W % 360, LON_E % 360
    lonm = (lon >= w) & (lon <= e) if w <= e else (lon >= w) | (lon <= e)
    return ds.isel(latitude=np.where(latm)[0], longitude=np.where(lonm)[0])


def _regional_mean(da):
    return da.mean(dim=["latitude", "longitude"]).to_series()


def build_features(start, end, out, source="wb2", test=False):
    import time as _time
    import xarray as xr
    print(f"opening {source} ...", flush=True)
    ds = _open(source)
    # keep ONLY the variables we need, before any read/compute (big speedup)
    needed = [v for v in (SINGLE_VARS + ARCO_EXTRA + list(PLEVELS)) if v in ds.data_vars]
    ds = ds[needed]
    print(f"  using {len(needed)} variables: {needed}", flush=True)
    if test:
        end = (pd.Timestamp(start) + pd.Timedelta(days=31)).strftime("%Y-%m-%d")
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    print(f"extracting {years.start}-{years.stop - 1} ...", flush=True)

    parts = []
    for yr in years:
        _t0 = _time.time()
        y0 = max(f"{yr}-01-01", start)
        y1 = min(f"{yr}-12-31", end)
        sub = ds.sel(time=slice(y0, y1))
        if "level" in sub.dims:                       # keep only needed levels early
            keep = sorted({l for ls in PLEVELS.values() for l in ls})
            sub = sub.sel(level=[l for l in keep if l in sub.level.values])
        box = _box(sub)
        daily = box.resample(time="1D").mean().compute()   # small after box-subset

        f = pd.DataFrame()
        for v in SINGLE_VARS + ARCO_EXTRA:
            if v in daily:
                f[f"{v}_mean"] = _regional_mean(daily[v])
        # 10 m wind speed
        if {"10m_u_component_of_wind", "10m_v_component_of_wind"} <= set(daily.data_vars):
            spd = np.sqrt(daily["10m_u_component_of_wind"]**2 +
                          daily["10m_v_component_of_wind"]**2)
            f["wind10_mean"] = _regional_mean(spd)
        # IVT magnitude (ARCO only)
        ve, vn = ("vertical_integral_of_eastward_water_vapour_flux",
                  "vertical_integral_of_northward_water_vapour_flux")
        if ve in daily and vn in daily:
            f["ivt_mean"] = _regional_mean(np.sqrt(daily[ve]**2 + daily[vn]**2))
        # MSLP gradients (proxy geostrophic flow)
        if "mean_sea_level_pressure" in daily:
            msl = daily["mean_sea_level_pressure"]
            f["msl_grad_ns"] = (msl.isel(latitude=0) - msl.isel(latitude=-1)
                                ).mean("longitude").to_series()
            f["msl_grad_ew"] = (msl.isel(longitude=-1) - msl.isel(longitude=0)
                                ).mean("latitude").to_series()
        # pressure-level means
        for v, levs in PLEVELS.items():
            if v in daily:
                for lev in levs:
                    if lev in daily.level.values:
                        f[f"{v}_{lev}_mean"] = _regional_mean(daily[v].sel(level=lev))
        parts.append(f)
        # incremental save so partial progress is never lost
        feats = pd.concat(parts).sort_index()
        feats.index.name = "date"
        (feats.to_parquet(out) if out.endswith(".parquet") else feats.to_csv(out))
        print(f"  {yr}: {f.shape[0]} days, {f.shape[1]} feats "
              f"({_time.time() - _t0:.0f}s)  [saved {feats.shape[0]} rows]", flush=True)

    print(f"\nDONE: saved {feats.shape} -> {out}\n{feats.head()}")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1989-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--out", default="era5_london_daily.parquet")
    ap.add_argument("--source", default="wb2", choices=list(SOURCES))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--area", default=None, help="N,W,S,E box (default London)")
    ap.add_argument("--list-vars", action="store_true")
    a = ap.parse_args()
    if a.area:
        LAT_N, LON_W, LAT_S, LON_E = [float(x) for x in a.area.split(",")]
    if a.list_vars:
        print("\n".join(sorted(map(str, _open(a.source).data_vars))))
    else:
        build_features(a.start, a.end, a.out, source=a.source, test=a.test)
