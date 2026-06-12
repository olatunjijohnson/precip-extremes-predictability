# Accompanying code — "How Predictable Are Rare Precipitation Extremes?"

This is the code package for the paper (`../paper/main.tex`): all models, the
evaluation protocol, the data-retrieval/processing scripts, and the figure/table
generators. Self-contained PyTorch + scikit-learn — **no gpytorch needed**.

## Files
| file | contents |
|------|----------|
| `tcdgp.py`    | the deep-GP–EVT hurdle: `FeatureExtractor` (deep kernel), `RBFKernel`, `CoregSVGP` (coregionalised sparse variational GP), `HurdleGPD` likelihood, `TCDGP`. |
| `distreg.py`  | zero-inflated distributional regression: log-normal, gamma, and **eGPD** positive parts (the models that win the bake-off). |
| `evaluation.py`| proper scores: Brier reliability/resolution decomposition, (tw)CRPS, CRPS-from-CDF, climatology/POT references, skill scores. |
| `conformal.py`| `split_conformal_upper` (+ GPD score-extrapolation) and `adaptive_conformal` (ACI). |
| `data.py`     | `load_real` (index + antecedent features, horizon-aware, leakage-safe), `load_drivers`, `make_synthetic`. |
| `teleconnections.py` | fetch/parse NAO, AO, Niño3.4 (NOAA). |
| `fetch_era5_arco.py` | fetch ERA5 circulation/moisture from WeatherBench2 zarr. |
| `fetch_era5_cds.py`  | fetch ERA5 convective-instability fields (CAPE/CIN/K-index/Total-Totals/IVT) from the Copernicus CDS. |
| `run_prototype.py` | all experiments: `model_bakeoff`, `calibration_eval`, `feature_screen`, `intensity_test`, `intensity_firmup`, `intensity_multisplit`, `driver_ablation`, `horizon_sweep`, synthetic validation. |
| `../paper/figures/*.py` | scripts that produce every figure in the paper. |

## Run
```bash
pip install torch numpy pandas scipy scikit-learn
python3 run_prototype.py            # CPU is fine; ~2–4 min
```

## What it demonstrates (validated)
**(1) Synthetic — borrowing strength.** Data generated with true ρ = 0.70.
- The **coupled** (dependent) hurdle recovers **ρ ≈ +0.8**; the independent
  model is fixed at 0.
- Tail log-likelihood improves dramatically from the **independent** model
  (whose tail estimate is unstable with few exceedances) to the **coupled**
  model — the borrowing-strength effect the paper claims (ablation #6).

**(2) Real — calibration.** London 95th-percentile index, train 1989–2008 /
test 2009–2018:
- ELBO converges; split-conformal and ACI both give **≈0.90 coverage at the
  0.90 target** (near-nominal).
- Occurrence PR-AUC sits just above the prevalence baseline — expected with
  index-only features and *no ERA5 yet* (adding the antecedent ERA5 drivers from
  `../02_era5_data.md` is the next lever).

> Numbers vary slightly by seed/hardware; exact figures from one run are logged
> in `../04_prototype_results.md`.

## Fetching the atmospheric drivers
```bash
python3 teleconnections.py        # NAO/AO/ENSO -> teleconnections_daily.csv (no account)
pip install xarray zarr gcsfs dask scikit-learn pyarrow
python3 fetch_era5_arco.py --test                 # 1-month check (ARCO cloud zarr)
python3 fetch_era5_arco.py --start 1989-01-01 --end 2018-12-31 \
        --out era5_london_daily.parquet
```
`run_prototype.py` auto-detects `era5_london_daily.parquet` and
`teleconnections_daily.csv` here and attaches them (lagged, antecedent); otherwise
it runs index-only. See `../02_era5_data.md` for both access routes.

## Status / next steps (this is a prototype, not the final model)
- [x] ERA5 + teleconnection fetch tooling (`fetch_era5_arco.py`, `teleconnections.py`);
      run them where you have internet, then re-run the prototype.
- [ ] Bulk distribution `F_blw` for full-distribution scoring (twCRPS over the
      whole line); currently the tail/occurrence are modelled, bulk is a crude
      placeholder in `predictive_cdf`.
- [ ] Minibatched natural-gradient / longer training; inducing-point init by
      k-means; ARD pruning.
- [ ] 2-layer deep GP variant (currently deep **kernel**) for the depth ablation.
- [ ] Full metric suite (twCRPS, reliability, BSS/CRPSS) + baselines wired to
      the experiment plan in `../01_method_and_experiments.md`.
- [ ] Appendix-grade proof for Proposition 1 (coverage).
