# How Predictable Are Rare Precipitation Extremes?

Code, paper, and processed data for *"How Predictable Are Rare Precipitation
Extremes? A Calibration-Aware Assessment for London"* (and Paris) — a rigorous,
calibration-aware predictability assessment of daily precipitation extremes.

**Private working repository** (MSc Data Science extended project; supervisor
Dr Olatunji Johnson, student Fankai Dong, University of Manchester).

## Layout
```
Paper/
  paper/            # manuscript: main.tex, supplementary.tex, references.bib, figures/
  prototype/        # all code (models, evaluation, data-fetch, figure scripts)
  *.md              # planning, method, results, and revision notes
data/               # processed precipitation-extreme indices (London, Paris)
```

## Code (`Paper/prototype/`)
| file | role |
|------|------|
| `tcdgp.py` | deep-GP–EVT hurdle (coregionalised sparse variational GP) |
| `distreg.py` | zero-inflated distributional regression (log-normal / gamma / eGPD) |
| `evaluation.py` | proper scores: Brier decomposition, (tw)CRPS, skill scores |
| `conformal.py` | tail-targeted + adaptive conformal calibration |
| `data.py` | index loader (horizon-aware, leakage-safe) + driver merge |
| `run_prototype.py` | all experiments: bake-off, feature screen (FDR), intensity tests (block bootstrap, multi-split), stability selection, GP sensitivity, horizon sweep, synthetic validation |
| `fetch_*.py` | data retrieval (city index, ERA5 WeatherBench2, ERA5 CDS, teleconnections) |

## Data
- **Indices** (`data/*_precip_extreme_index_*.csv`): the standardised
  precipitation-extreme index, daily 1989–2018, derived from the Copernicus CDS
  product *Extreme precipitation risk indicators for European cities*
  (`sis-european-risk-extreme-precipitation-indicators`). Reproduce with
  `prototype/fetch_city_index.py`.
- **Processed ERA5 driver tables** (`prototype/era5_*.parquet`,
  `teleconnections_daily.csv`): antecedent atmospheric predictors. Reproduce
  with `prototype/fetch_era5_*.py`.
- The raw 376 MB Copernicus netCDF archive is **not** committed; download from
  the CDS source above.

## Reproduce
```bash
cd Paper/prototype
pip install torch numpy pandas scipy scikit-learn xgboost
python3 run_prototype.py          # full experiment suite (London)
```
Atmospheric-driver fetching needs `xarray zarr gcsfs dask cdsapi` and a
Copernicus CDS account (`~/.cdsapirc`); see `Paper/02_era5_data.md`.

## Build the paper
```bash
cd Paper/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
