# How Predictable Are Rare Precipitation Extremes?

### A calibration-aware assessment for London and Paris

**Olatunji Johnson and Isqeel Ogunsola** — Department of Mathematics, University of Manchester, UK

> **Status:** submitted to *Environmetrics* (2026).

This repository contains the manuscript, supplementary material, and complete code
for a study of how predictable rare daily precipitation extremes are, and how that
predictability should be measured. Using thirty years (1989–2018) of a standardised
precipitation-extreme index for London and Paris with strictly antecedent ERA5
drivers, we assemble a calibration-aware evaluation protocol (proper scores, skill
against climatology, bootstrap intervals, multi-split robustness) and run a bake-off
from logistic regression to a Gaussian-process extreme-value hurdle.

**Key findings.** A Gaussian-process extreme-value hurdle is the strongest
full-distribution forecaster on the tail, while simple calibrated models match it on
occurrence, and the choice of response distribution matters as much as model class.
Occurrence and the magnitude of the largest events are weakly but genuinely
predictable from antecedent convective instability (CAPE/CIN), with a CRPS skill over
climatology that excludes zero across four independent splits in each city. Conformal
coverage, though guaranteed by construction, is shown to be insufficient on its own as
a measure of forecast quality.

## Layout

```
Paper/
  paper/        manuscript: main.tex, supplementary.tex, references.bib, figures/
  prototype/    code: models, evaluation, data retrieval, figure/table generators
data/           processed precipitation-extreme indices (London, Paris)
```

See `Paper/prototype/README.md` for a file-by-file guide to the code.

## Reproduce

```bash
cd Paper/prototype
pip install torch numpy pandas scipy scikit-learn xgboost
python3 run_prototype.py          # main experiments (CPU is fine; a few minutes)
python3 revision_analyses.py all  # supplementary analyses
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data availability

The processed daily precipitation-extreme indices (`data/`) and the antecedent
feature tables (`Paper/prototype/`) are derived from public-domain resources: the
Copernicus Climate Data Store *Extreme precipitation risk indicators for Europe and
European cities (1950–2019)* (https://cds.climate.copernicus.eu); ERA5 reanalysis
(Copernicus CDS `reanalysis-era5-single-levels`) and the WeatherBench2 regridded ERA5
archive (https://weatherbench2.readthedocs.io); and the NOAA teleconnection indices
(NOAA CPC/PSL, https://www.cpc.ncep.noaa.gov).
