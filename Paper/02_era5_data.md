# ERA5 Data Recipe — antecedent atmospheric drivers

Goal: build a daily table of **strictly antecedent** atmospheric predictors aligned to the existing
London index (1989–2018), to add to the feature matrix in `01_method_and_experiments.md`.

> **Leakage rule (non-negotiable).** Every ERA5 feature must be lagged so it is known **before**
> the forecast time. For a horizon-$h$ forecast of day $t+h$, use drivers from day $t$ or earlier.
> Never use concurrent/future fields (ERA5 is reanalysis → concurrent use is circular). In
> particular, do **not** use ERA5 total precipitation at or after the target day as a predictor.

---

## 0. Two routes (use ARCO for bulk)

There are two ways to obtain the ERA5 driver fields. **For 30 years over a London box, the
cloud-zarr route is far easier** — no account, no licence dance, no download queue.

| | **Route A — ARCO-ERA5 (recommended)** | **Route B — Copernicus CDS (official)** |
|---|---|---|
| Account | none | ECMWF/CDS account + per-dataset licence |
| Access | public Google Cloud zarr via `xarray` | `cdsapi` requests, server-side queue |
| Speed | slice & go | queue can be slow for 30 yr |
| Use when | bulk extraction, ML workflows | canonical provenance / variables ARCO lacks |
| Script | **`prototype/fetch_era5_arco.py`** | §4 below (`cdsapi`) |

**Quick win first (no account, 5 min):** the teleconnection indices (NAO/AO/ENSO) are tiny text
files — fetch them with **`prototype/teleconnections.py`** and run the first covariate ablation while
the ERA5 fields download.

### Route A setup (ARCO-ERA5)
```bash
pip install xarray zarr gcsfs dask scikit-learn pandas numpy pyarrow
python3 prototype/fetch_era5_arco.py --list-vars      # inspect variable names
python3 prototype/fetch_era5_arco.py --test           # 1-month smoke test
python3 prototype/fetch_era5_arco.py --start 1989-01-01 --end 2018-12-31 \
        --out prototype/era5_london_daily.parquet
```
The analysis-ready zarr is `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
(confirm the current path in the [arco-era5 README](https://github.com/google-research/arco-era5)).
Gotchas the script already handles: ARCO longitude is **0–360°** (the London/Atlantic box wraps
across 0°) and latitude is **descending**.

### Route B setup (Copernicus CDS — current, post-2024 system)
1. Register at `cds.climate.copernicus.eu` (ECMWF single sign-on).
2. Copy your **Personal Access Token** from `cds.climate.copernicus.eu/profile`.
3. Create `~/.cdsapirc` (the **new** endpoint; the old `.../api/v2` + `UID:KEY` is legacy):
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your Personal Access Token>
   ```
4. `pip install "cdsapi>=0.7" xarray netCDF4 cfgrib`.
5. **Accept the licence on each dataset's page** (bottom of the download form) or the API fails silently.
6. Prefer the **derived daily-statistics** product (`derived-era5-single-levels-daily-statistics`)
   to skip hourly volume. Test a one-month, one-variable request before the bulk loop.

---

## 2. Variables to pull (and why)

**Single-level fields** (`reanalysis-era5-single-levels`):
| Variable | CDS short name | Why it matters for UK rainfall extremes |
|---|---|---|
| Mean sea-level pressure | `msl` | synoptic setup, low-pressure systems |
| Total column water vapour | `tcwv` | moisture availability |
| Vertical integral of eastward water vapour flux | `viwve` | IVT (atmospheric rivers) — east component |
| Vertical integral of northward water vapour flux | `viwvn` | IVT — north component |
| CAPE | `cape` | convective potential |
| 2 m temperature | `2t` | thermodynamic context |
| 10 m u / v wind | `10u`, `10v` | low-level flow |

**Pressure-level fields** (`reanalysis-era5-pressure-levels`, levels 500 & 850 hPa):
| Variable | short name | Why |
|---|---|---|
| Geopotential (Z500, Z850) | `z` | large-scale circulation / ridges & troughs |
| Specific humidity (850) | `q` | low-level moisture |
| u / v winds (250 & 500) | `u`,`v` | jet-stream position / steering |

**Derived:** IVT magnitude $= \sqrt{\texttt{viwve}^2 + \texttt{viwvn}^2}$.

---

## 3. Domain, resolution, period

- **Spatial box** (capture the synoptic environment *upstream*, not just London):
  roughly **45–60° N, 20° W–10° E** (`area = [60, -20, 45, 10]` as `[N, W, S, E]`), 0.25° grid.
- **Period:** 1989-01-01 to 2018-12-31, to match the index.
- **Time:** request hourly then aggregate to daily — **or** use the CDS derived
  *daily-statistics* application (`derived-era5-single-levels-daily-statistics`) to skip hourly
  downloads (recommended; far less data). Verify availability for pressure levels.

---

## 4. Sample download script (single-level, one year — loop over years)

```python
import cdsapi
c = cdsapi.Client()
for year in range(1989, 2019):
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "mean_sea_level_pressure", "total_column_water_vapour",
                "vertical_integral_of_eastward_water_vapour_flux",
                "vertical_integral_of_northward_water_vapour_flux",
                "convective_available_potential_energy",
                "2m_temperature", "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(0, 24, 6)],  # 6-hourly is enough
            "area":  [60, -20, 45, 10],   # N, W, S, E
        },
        f"era5_sl_{year}.nc",
    )
```

---

## 5. Processing to a daily feature table

```python
import xarray as xr, numpy as np, pandas as pd

ds = xr.open_mfdataset("era5_sl_*.nc")           # all years
# 1) aggregate to daily (mean and, for some vars, max)
daily = ds.resample(time="1D").mean()
daily_max = ds[["cape", "tcwv"]].resample(time="1D").max()

# 2) derived: IVT magnitude
ivt = np.sqrt(daily["viwve"]**2 + daily["viwvn"]**2)

# 3) reduce the spatial field to features. Options (do BOTH, compare):
#    (a) regional means / gradients over the box
#    (b) leading EOFs / PCs of each field (capture the dominant patterns)
feat = pd.DataFrame(index=daily["time"].values)
feat["msl_mean"]  = daily["msl"].mean(dim=["latitude","longitude"]).values
feat["tcwv_mean"] = daily["tcwv"].mean(dim=["latitude","longitude"]).values
feat["ivt_mean"]  = ivt.mean(dim=["latitude","longitude"]).values
# ... add Z500 mean/gradient, q850, jet-level winds, EOF scores ...

# 4) LAG every feature (antecedent only!). For horizon h, shift by h days.
feat_lagged = feat.shift(1)        # at minimum lag 1 day; build per-horizon as needed
```

Merge `feat_lagged` onto the index table by date, then feed into `build_features()`.

> **Already implemented.** `prototype/fetch_era5_arco.py` performs steps 1–3 (daily aggregation,
> IVT, regional means + MSLP gradients + Z500 EOFs) and writes the parquet. Lagging (step 4) is done
> by `prototype/data.py::load_real(..., drivers_df=..., driver_lag=1)`, which keeps everything
> antecedent.

---

## 6. Teleconnection indices (cheap, do these first)

Implemented in **`prototype/teleconnections.py`** — run `python3 prototype/teleconnections.py` to
download, parse and cache `teleconnections_daily.csv`. Verified machine-readable sources/formats:

| Index | Source | Format |
|---|---|---|
| **NAO** | `psl.noaa.gov/data/correlation/nao.data` | PSL: `year` + 12 monthly values, sentinel `-99.90` |
| **Niño3.4** (ENSO) | `psl.noaa.gov/data/correlation/nina34.data` | same PSL format |
| **AO** | CPC `monthly.ao.index.b50.current.ascii` | `year month value` |
| ONI (alt ENSO) | CPC `data/indices/oni.ascii.txt` | `SEAS YR TOTAL ANOM` |

These are monthly indices (the daily CPC files are unstable and noisy); the loader broadcasts them
to daily and `data.py` lags them. A useful first covariate-ablation step before the full ERA5 pull.

### Wiring into the model
Once fetched, `prototype/run_prototype.py` **auto-detects** `era5_london_daily.parquet` and
`teleconnections_daily.csv` in the prototype folder and attaches them; otherwise it runs index-only.

---

## 7. Storage / compute notes
- 6-hourly single-level over the regional box, 30 yr: order a few GB raw; much smaller after daily
  aggregation. Pressure levels add more — restrict to 500/850 (+250 for jet) only.
- The CDS **queue** is the real bottleneck; submit year-by-year jobs (script above) and/or use the
  daily-statistics product to cut volume.
- Cache the processed daily feature table (`era5_features_daily.parquet`) so modelling never
  re-reads raw netCDF.

---

## 8. Checklist
- [ ] CDS account + licences accepted; `.cdsapirc` configured.
- [ ] Confirm current dataset IDs / variable names in the CDS API tab.
- [ ] Teleconnection indices (NAO/AO/ENSO) downloaded and lagged (quick win).
- [ ] ERA5 single-level fields downloaded (1989–2018, regional box).
- [ ] ERA5 pressure levels (500/850, +250 winds) downloaded.
- [ ] Daily aggregation + IVT + spatial reduction (means/gradients/EOFs).
- [ ] All features lagged (antecedent only); leakage audit done.
- [ ] Processed table cached as parquet and merged to the index.
