# Data included in this repository

The repository contains the processed daily inputs used by the analysis. Dates
cover 1989–2018.

| File | Contents |
|---|---|
| `london_precip_extreme_index_95th_1989_2018.csv` | London city-maximum precipitation-extreme index based on the cell-specific 95th percentile |
| `paris_precip_extreme_index_95th_1989_2018.csv` | Paris equivalent |
| `london_precip_extreme_index_99th_1989_2018.csv` | London 99th-percentile sensitivity series |
| `era5_london_daily.parquet`, `era5_paris_daily.parquet` | Daily ERA5 circulation and moisture summaries |
| `era5_cape_ivt_daily.parquet`, `era5_paris_cape_ivt_daily.parquet` | Daily convective and integrated-moisture predictors |
| `teleconnections_daily.csv` | Daily table of monthly teleconnection values; the modelling code applies a one-month lag |

The precipitation index is derived from the Copernicus Climate Data Store
product *Extreme precipitation risk indicators for Europe and European cities
(1950–2019)*. Atmospheric predictors come from ERA5, including the
WeatherBench2 regridded archive, and teleconnection indices come from NOAA.
The retrieval scripts in `code/` document the processing route.

These are processed research data, not operational forecasts. The analysis
uses observations up to 2008 for model development and 2009–2018 for the main
out-of-sample evaluation.
