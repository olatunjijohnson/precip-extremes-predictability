# Accompanying code — *How Predictable Are Rare Precipitation Extremes?*

Code package for the paper (`../paper/main.tex`): all models, the evaluation
protocol, the data-retrieval/processing scripts, and the figure/table generators.
Self-contained PyTorch + scikit-learn — **no gpytorch needed**.

## Core modules
| file | contents |
|------|----------|
| `tcdgp.py`    | the **GP–EVT hurdle** (`TCDGP`): a coregionalised sparse variational Gaussian process with a Generalised-Pareto hurdle likelihood (`RBFKernel`, `CoregSVGP`, `HurdleGPD`). The paper uses the plain single-layer variant (`deep=False`); a deep-kernel option (`deep=True`) is available and reported only as a depth ablation. |
| `distreg.py`  | zero-inflated distributional regressions: log-normal, gamma, and **eGPD** positive parts. |
| `evaluation.py`| proper scores: Brier reliability/resolution decomposition, (tw)CRPS, CRPS-from-CDF, climatology/POT references, skill scores. |
| `conformal.py`| `split_conformal_upper` (+ GPD score-extrapolation) and `adaptive_conformal` (ACI). |
| `data.py`     | `load_real` (index + antecedent features, horizon-aware, leakage-safe), `load_drivers`, `make_synthetic`. |

## Experiment drivers
| file | contents |
|------|----------|
| `run_prototype.py` | the main experiments: `model_bakeoff` (Table 2), `calibration_eval` (reliability diagrams), `feature_screen`, `intensity_test`/`intensity_firmup`/`intensity_multisplit` (hold-out intensity), `driver_ablation`, `horizon_sweep`. Set the city with `set_city("london"|"paris")`. |
| `revision_analyses.py` | supplementary analyses: (A) threshold-`u` sensitivity, (B) bootstrap block-size, (C) synthetic coupling recovery, (D) real-data `ρ̂`, (E) feature-importance table (Table S2), (F) horizon-wise BSS table (Table S3), (G) 99th-percentile bake-off (Table S1). |
| `compute_gp_confidence_intervals.py` | moving-block bootstrap CIs for the GP–EVT hurdle's BSS/twCRPS (both cities). |
| `compare_coupled_vs_independent.py`  | coupled vs independent GP–EVT hurdle (Supplementary S4 — coupling is empirically inert). |
| `generate_horizon_json.py` | seed-averaged horizon sweep for both cities → `horizon_london.json`, `horizon_paris.json` (inputs to the horizon figure). |

## Data-retrieval scripts
| file | contents |
|------|----------|
| `teleconnections.py` | fetch/parse NAO, AO, Niño-3.4 (NOAA) → `teleconnections_daily.csv`. |
| `fetch_era5_arco.py` | ERA5 circulation/moisture from the WeatherBench2 zarr archive. |
| `fetch_era5_cds.py`  | ERA5 convective-instability fields (CAPE/CIN/K-index/Total-Totals/IVT) from the Copernicus CDS. |
| `fetch_city_index.py`| retrieve the city precipitation-extreme index. |

Figures are produced by `../paper/figures/*.py`. The cached inputs they read
(`reliability*.npz`, `horizon_*.json`, the ERA5 `*.parquet`, `teleconnections_daily.csv`)
are included here, so the figures rebuild without re-running the full pipeline.

## Run
```bash
pip install torch numpy pandas scipy scikit-learn xgboost
python3 run_prototype.py            # CPU is fine; a few minutes
python3 revision_analyses.py all    # supplementary numbers (C/D take a few min each)
```

## Fetching the atmospheric drivers (optional — cached copies are included)
```bash
python3 teleconnections.py          # NAO/AO/ENSO (no account needed)
pip install xarray zarr gcsfs dask pyarrow cdsapi
python3 fetch_era5_arco.py --start 1989-01-01 --end 2018-12-31 --out era5_london_daily.parquet
python3 fetch_era5_cds.py           # needs a Copernicus CDS API key
```
`run_prototype.py` auto-detects the driver files here and attaches them
(strictly lagged / antecedent); otherwise it runs index-only.
