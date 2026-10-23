# Forecast attribution for precipitation extremes

Reproducible code and processed data for a retrospective study of one-day-ahead
predictive skill in city-scale precipitation-extreme indices for London and
Paris. The repository separates skill due to seasonal variation, recent index
history, atmospheric predictor groups, event occurrence, and precipitation
amount above the extreme threshold.

The analysis compares seasonal forecasts, regularised logistic and linear
hurdle models, and a sparse variational Gaussian-process extreme-value hurdle.
All principal comparisons use the same test days, paired score differences,
95% moving-block bootstrap intervals, and predictors available before the
target day.

This repository contains code and data only. The manuscript is intentionally
not included.

## Main findings represented by the archived outputs

- Accounting for seasonality is important, particularly for Paris.
- Circulation and moisture predictors provide the clearest additional signal;
  gains from teleconnections and convective predictors depend on the specified
  sequence and are less consistent.
- A regularised linear hurdle captures most of the available skill. The more
  flexible Gaussian-process hurdle has no clear advantage in London and a
  positive upper-tail score difference in Paris.
- Controlled comparisons show that the Paris improvement is mainly associated
  with event-occurrence probabilities. Evidence for improved prediction of the
  amount above the threshold is weaker.

These conclusions concern retrospective predictability within a reanalysis-
based system, not the performance of an operational numerical weather forecast.

## Repository layout

```text
code/                  models, scoring rules, analyses, and retrieval scripts
code/results/          archived machine-readable outputs from the final analysis
data/                  processed precipitation indices and atmospheric predictors
requirements.txt       Python dependencies
```

The central modules are:

| File | Purpose |
|---|---|
| `code/data.py` | Constructs lagged index features and joins antecedent atmospheric predictors |
| `code/evaluation.py` | Brier score, threshold-weighted CRPS, decompositions, and skill scores |
| `code/tcdgp.py` | Sparse variational Gaussian-process GPD hurdle |
| `code/major_revision_analysis.py` | Seasonal baselines, nested predictor stages, model comparisons, and conditional-amount scores |
| `code/second_review_analyses.py` | Exact tail-score calculations, controlled occurrence/amount combinations, and sensitivity analyses |
| `code/corrected_feature_gp_analysis.py` | Final 48-predictor, 3-by-3 occurrence-and-amount factorial comparison |
| `code/predictor_group_sensitivity.py` | Alternative-order and drop-one-group analyses |
| `code/corrected_gp_structure_sensitivity.py` | Kernel, coregionalisation, inducing-point, and seed checks |

Files containing `corrected` use the final 48-predictor grouping, with
total-column water vapour and integrated vapour transport assigned to the
circulation-and-moisture group and the duplicated water-vapour feature removed.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core analysis does not require `gpytorch`.

## Reproduce the final analyses

Run commands from the repository root. The scripts write CSV, JSON, NPZ, and
PDF outputs to `code/results/`.

```bash
cd code

# Core baselines, predictor stages, and model comparisons
python major_revision_analysis.py

# Controlled occurrence-versus-amount comparisons and score sensitivities
python second_review_analyses.py

# Final predictor grouping and occurrence-stage results
python corrected_occurrence_stages.py
python corrected_feature_gp_analysis.py
python predictor_group_sensitivity.py

# Diagnostics and robustness checks
python corrected_feature_postchecks.py
python corrected_gp_structure_sensitivity.py
python gp_posterior_mixture_check.py
python gp_endpoint_diagnostics.py
python evt_diagnostics.py
python intensity_recheck.py
python make_corrected_reliability_figure.py
```

Gaussian-process scripts are the slowest part of the workflow and can take
several hours on a CPU. Random seeds, bootstrap settings, penalties, and the
1989–2008/2009–2018 split are recorded in the scripts and JSON output files.
The outputs used for the SERRA submission are preserved in the GitHub release
tagged `serra-submission-v1.0`.

## Data provenance and timing

The processed inputs are included in `data/`; see `data/README.md`. Monthly
teleconnection indices are shifted by one calendar month. Cached daily ERA5
aggregates are shifted by an additional day, so the main atmospheric predictors
for target day *t* are from *t*−2. The index-history variables use information
through *t*−1. This conservative timing avoids using target-day information but
means the atmospheric component is based on older information than the latest
index value.

The retrieval scripts are supplied for provenance. Re-downloading ERA5 requires
network access and, for Copernicus data, configured CDS API credentials.
