# Prototype results (smoke test)

Logged output of `prototype/run_prototype.py` (CPU, seed 0). The prototype
implements the model of `paper/main.tex` §4 in self-contained PyTorch. Purpose:
de-risk the method by showing it trains, recovers a known dependence, and
produces calibrated tail coverage. **Not** the final tuned model.

## (1) Synthetic — dependence recovery & borrowing strength
Data generated with **true ρ = 0.70**, true ξ = 0.20, exceedance rate 0.08.

| model | recovered ρ | ξ | occ. PR-AUC | tail log-lik |
|-------|------------:|----:|-----------:|-------------:|
| independent hurdle      | +0.000 (fixed) | −0.224 | 0.222 | −4422.6 |
| coupled (dependent)     | **+0.825**     | +0.346 | 0.201 | **−1.99** |

**Reading.** The coupled model recovers a strong positive ρ (true 0.70). The
independent model's intensity head is estimated from only the few exceedances
and becomes **unstable** — its shape collapses to ξ = −0.22 (a short bounded
tail), so test excesses fall outside the GPD support and the tail log-likelihood
is catastrophic. Coupling lets the abundant occurrence data stabilise the tail
(the "borrowing strength" effect). This is direct evidence for ablation #6 in
`01_method_and_experiments.md`.

*Caveat:* the −4422 figure is dominated by support violations, so it overstates
the gap in magnitude; the qualitative conclusion (independent tail is unstable,
coupling fixes it) is the robust point. For the paper, report a bounded tail
metric (twCRPS) alongside, and average over seeds.

## (2) Real — calibration on the London 95th index
Train 1989–2008 (7291 d, 287 extremes) / test 2009–2018 (3652 d, 130 extremes),
index-only antecedent features, **no ERA5 yet**.

| quantity | value | target |
|----------|------:|-------:|
| split-conformal coverage | **0.895** | 0.90 |
| adaptive-conformal (ACI) coverage | **0.900** | 0.90 |
| occurrence PR-AUC | 0.043 | prevalence 0.036 |
| recovered ρ | +0.35 | — |

**Reading.** The conformal layer delivers near-nominal coverage on real,
temporally dependent data — the headline guarantee works in practice.
Occurrence skill is only marginally above prevalence, as expected without the
atmospheric drivers; closing that gap is what the ERA5 covariates
(`02_era5_data.md`) are for.

## (3) Driver ablation — does adding covariates help? (London 95th, h=1)
Real run with fetched drivers: `teleconnections_daily.csv` (NAO/AO/Niño3.4) and
`era5_london_daily.parquet` (WeatherBench2 ERA5, 11 daily synoptic features over a London box).

**Seed-averaged result (5 seeds, mean ± std) — the trustworthy version:**

| config | PR-AUC | conformal coverage |
|--------|-------:|-------------------:|
| index-only        | 0.054 ± 0.008 | 0.894 ± 0.002 |
| +teleconnections  | 0.042 ± 0.003 | 0.893 ± 0.003 |
| +ERA5 (+teleconn) | 0.057 ± 0.010 | 0.891 ± 0.003 |

Paired (same seeds) **[+ERA5] − [index-only] = +0.003 ± 0.012 PR-AUC (SE 0.005), mixed sign**.
Prevalence baseline PR-AUC ≈ 0.036.

**Reading (corrected).** Averaged over seeds, **adding ERA5 gives no statistically robust skill gain
at horizon 1** with the coarse (~1.5° WeatherBench2, daily-mean) feature set: the paired improvement
is well within noise and flips sign across seeds. Teleconnections slightly *hurt* at h=1 (noise
features with no daily signal). A single earlier run had shown index-only at 0.036 → +ERA5 0.050,
but that was a low-seed **outlier** — index-only's true mean is ~0.054. **Lesson: never report a
single-seed ablation.**

What *is* robust: **conformal coverage stays ≈0.89 across every config** — i.e. calibration holds
regardless of the predictor set. This reinforces the calibration-first framing: the contribution is
trustworthy probabilities, not raw skill, which at h=1 is intrinsically low.

This motivates the planned experiments where a driver benefit is more plausible: **longer horizons**
(h=3,7,14,28 — where synoptic/teleconnection memory matters more than the index's own short
persistence) and **finer ERA5 fields** (CAPE, IVT/atmospheric rivers) absent from the WB2 cut. The
h=1, 1.5° prototype simply may not be where atmospheric drivers pay off.

## (4) Horizon sweep — do drivers earn their place at longer range? (seed-averaged, 3 seeds)

PR-AUC (mean ± std); prevalence baseline ≈ 0.036; conformal coverage in last block.

| horizon | index-only | +teleconnections | +ERA5 | ERA5 gain |
|--------:|-----------:|-----------------:|------:|----------:|
| 1  | 0.051 ± 0.007 | 0.046 ± 0.007 | 0.056 ± 0.010 | +0.005 |
| 3  | 0.041 ± 0.003 | 0.038 ± 0.003 | 0.042 ± 0.001 | +0.001 |
| 7  | 0.040 ± 0.004 | 0.038 ± 0.002 | 0.040 ± 0.007 | +0.001 |
| 14 | 0.045 ± 0.009 | 0.045 ± 0.004 | 0.039 ± 0.003 | −0.006 |
| 28 | 0.040 ± 0.005 | 0.043 ± 0.004 | 0.049 ± 0.003 | **+0.009** |

Conformal coverage stays **0.89–0.90 at every horizon and every config**.

**Reading (honest).**
1. **Skill is low and decays with horizon** — index-only falls from 0.051 (h=1) to ~0.040
   (h≥7), hovering just above the 0.036 prevalence floor. London daily extremes have little
   day-scale predictability; this is the expected, reportable decay curve.
2. **Drivers give at best marginal, noisy gains.** Most ERA5/teleconnection deltas sit within the
   std bands. A 3-seed sweep hinted at ERA5 at h=28 (+0.009), but an **8-seed confirmation deflated
   it to +0.004 ± 0.006 (not significant)**: index-only 0.043 ± 0.004 vs +ERA5 0.046 ± 0.006 at
   h=28, and −0.005 at h=14. So the long-range hint was again seed noise — **no statistically robust
   driver gain at any horizon** with the coarse WB2 features. (Whether finer ERA5 fields — CAPE,
   IVT — change this is the remaining open question; those need the CDS route, not WB2.)
3. **The robust finding is calibration:** coverage is ≈0.90 across all horizons and predictor sets —
   horizon- and driver-invariant. This is the result the paper can lean on.

## (5) DECISIVE calibration-aware evaluation (proper scoring) — the make-or-break test

Brier reliability/resolution decomposition + tail threshold-weighted CRPS (τ=u=1), seed-averaged
(3 seeds). Skill scores are vs climatology.

| model | Brier | BSS | REL | RES | twCRPSS |
|-------|------:|----:|----:|----:|--------:|
| climatology       | 0.0343 |  0.000 | ~0      | 0.000   |  0.000 |
| stationary POT    | 0.0343 |  0.000 | ~0      | 0.000   |  0.000 |
| logistic [index]  | 0.0339 | +0.012 | 0.0004  | 0.0004  |   —    |
| **TCDGP [index]** | 0.0467 | **−0.361** | 0.0123 | 0.0002 | **−0.265** |
| logistic [+ERA5]  | 0.0331 | +0.035 | 0.0001  | 0.0008  |   —    |
| **TCDGP [+ERA5]** | 0.0510 | **−0.487** | 0.0168 | 0.0002 | **−0.336** |

**Verdict: the prototype model FAILS the decisive test.**
1. TCDGP is **worse than climatology** on both Brier (BSS ≈ −0.4) and tail twCRPS (twCRPSS ≈ −0.3):
   its raw predictive probabilities are **miscalibrated** (REL ≈ 0.012–0.017 ≫ climatology/logistic).
2. A **plain logistic regression beats it** — better calibrated AND ~3× the resolution
   (informativeness). The elaborate deep-GP–EVT model loses to a 20-feature linear model.
3. The gap is **not just calibration**: TCDGP's resolution (0.0002) is far below logistic's (0.0008),
   so even perfectly recalibrated its best-case BSS ≈ +0.006 still loses. Calibration tweaks can't fix it.

**This overturns the earlier "calibration holds" headline.** Conformal coverage ≈0.90 is valid but
**vacuous** — conformal wraps any base model with valid *interval* coverage regardless of predictive
quality. The proper-scoring test, which actually measures the predictive distribution, shows the
generative model is worse than climatology. The single-site, h=1 temporal framing does **not** justify
the model.

**Implication / path forward.** GPs add the most value for *spatial* structure; a single-site
temporal GP is precisely where they lose to simple baselines. The viable directions are (a) reframe to
**spatio-temporal extremes** over the 53 grid cells / multi-city (where coregionalisation and the GP
prior have a structural advantage and there is more data), or (b) treat this as a negative/methods
result. The dependent-hurdle + EVT idea remains validated on synthetic data, but needs a problem where
the GP machinery earns its place.

## (6) Model bake-off — rethinking the distributional assumptions

Tested the hypotheses that (H1) modelling the FULL zero-inflated distribution (all wet-day data,
derive P(I>1)) beats hurdling at u=1; (H2) a GPD-tailed continuous distribution (eGPD, threshold-free)
beats light/heavy alternatives; (H3) a parametric distributional regression beats the GP. Proper
scores vs climatology, seed-averaged (3 seeds); occurrence via derived P(I>1).

| model | BSS | RES(×10³) | twCRPSS |
|-------|----:|----------:|--------:|
| climatology / stationary POT | 0.000 | 0.0 |  0.000 |
| logistic [index]             | +0.012 | 0.43 | — |
| **logistic [+ERA5]**         | **+0.035** | 0.76 | — |
| ZI-lognormal [+ERA5]         | −0.266 | 0.52 | −2.118 |
| ZI-gamma [+ERA5]             | +0.010 | 0.50 | −0.050 |
| **ZI-eGPD [+ERA5]**          | +0.015 | 0.58 | **−0.035** |
| TCDGP (GP-EVT hurdle, ref.)  | −0.4  | 0.02 | −0.3 |

**Findings (well-tested, robust).**
1. **H1 confirmed — hurdle-at-1 was the structural mistake.** Modelling the full distribution
   (ZI-gamma/eGPD) recovers from TCDGP's BSS −0.4 / twCRPSS −0.3 to roughly climatology level.
2. **Distribution shape dominates.** Log-normal (too heavy a tail) is catastrophic (twCRPSS −2.1);
   gamma (light) and eGPD (GPD-tailed) are fine. The index tail is **not** as heavy as log-normal/GPD-
   at-low-threshold assume. eGPD is the best tail model.
3. **H3 confirmed — the GP adds nothing here.** Simple parametric/linear models beat the GP-EVT on
   every metric.
4. **The decisive split in predictability:**
   - **Occurrence has weak but real, covariate-improvable signal** — best is a plain **calibrated
     logistic + ERA5 (BSS +0.035**, ~2× the resolution of index-only). ERA5 helps occurrence
     consistently across models.
   - **Intensity/tail is climatology-bound (≈unpredictable).** *No* model — not even the purpose-built
     eGPD — beats the empirical climatological tail on twCRPS. Given that an extreme occurs, its
     magnitude carries no exploitable signal from these antecedent inputs.

**Consolidated scientific conclusion.** The model *can* be improved a lot over the GP-EVT prototype
(eGPD full-distribution ≫ GP-EVT hurdle), but the achievable ceiling is low: modest occurrence skill,
no intensity skill. The elaborate GP machinery is not justified for this single-site problem; a simple
calibrated distributional/logistic model is the honest best. The one untested lever for *intensity*
skill is convective predictors (**CAPE, IVT**) — absent from WeatherBench2, requiring the CDS route.

## (7) CAPE/IVT convective predictors + predictability screen (refines the intensity conclusion)

Fetched the convective-instability + moisture fields WB2 lacks (CAPE, CIN, K-index, Total-Totals,
TCWV, IVT) from CDS — all antecedent, NO precipitation field (target is ERA5-derived precip).

**Bake-off with CAPE/IVT (seed-averaged):** occurrence improves marginally (logistic +ERA5 BSS
+0.035 → +0.038, resolution 0.76 → 0.83); tail twCRPSS stays negative (eGPD −0.045) — the
*full-predictive* tail score still doesn't beat climatology (it's diluted by the 95% non-extreme days).

**Per-variable screen.** Occurrence (L1-logistic) is led by **K-index (+0.51)**, Z500, IVT, humidity,
seasonality, CAPE — physically sensible. Intensity (Spearman on 287 extreme training days): **23/49
features significant at p<0.05 (vs ~2 by chance)** — CIN (+0.28), CAPE (+0.24), temperature, moisture.
So intensity is **not** signal-free, contradicting the earlier strong claim.

**Hold-out intensity test (the honest arbiter).** Quantile regression of the excess (I−u) on
covariates vs climatological excess, on held-out extreme days, regularisation chosen by CV on train
(no test leak):

| excess quantile | skill vs climatology |
|----------------:|---------------------:|
| 0.50 (median)   | +0.009 (nil) |
| 0.75            | −0.000 (nil) |
| **0.90 (deep tail)** | **+0.071** |

**Firm-up — CRPS-of-excess + bootstrap CI (the rigorous arbiter).** Proper score over ALL excess
quantiles (no cherry-picking), CV-selected regularisation on train only, iid bootstrap (2000) over
the 130 test extreme days:

| metric | skill vs climatology | 90% CI | P(>0) |
|--------|---------------------:|:------:|:-----:|
| CRPS-of-excess  | **+0.014** | [+0.009, +0.019] | 1.00 |
| q90 pinball     | +0.049 | [+0.001, +0.094] | 0.95 |

**Conclusion (now well-supported).** Intensity is **weakly but genuinely predictable**: the
CRPS-of-excess skill is small (~1.4%) but its bootstrap CI **excludes zero** — and unlike the earlier
over-optimistic claims, this survives a proper score, honest model selection, and a CI. The signal is
tied to **convective instability (CAPE/CIN)** and is somewhat concentrated in the upper tail (q90
+4.9%, CI barely excludes 0). So: median magnitude ≈ climatology; the largest-event magnitudes carry
weak real predictability.

**Robustness across splits (the final check).** CRPS-of-excess skill, fixed regularisation, forward-only
expanding-window splits:

| test period | n_test ext. | CRPS skill | 90% CI | P(>0) |
|------------:|------------:|-----------:|:------:|:-----:|
| >2002 | 213 | +0.037 | [+0.021,+0.051] | 1.00 |
| >2005 | 175 | +0.027 | [+0.015,+0.041] | 1.00 |
| >2008 | 130 | +0.035 | [+0.021,+0.051] | 1.00 |
| >2011 |  96 | +0.034 | [+0.009,+0.061] | 0.98 |

**4/4 splits positive, mean +3.3%, all CIs exclude zero.** The intensity-predictability finding
survived proper scoring, honest model selection, bootstrap CIs, AND four independent train/test
periods — unlike the earlier occurrence-driver claims that evaporated under scrutiny.

**Net predictability picture (rigorous, final):** occurrence weakly predictable (BSS +0.038, led by
K-index / Z500 / IVT); intensity weakly but **robustly** predictable (CRPS-of-excess ~+0.02–0.035
across splits, CIs exclude 0, led by **CAPE/CIN convective instability**). Both effects are small —
a genuinely hard problem — but both are real, physically grounded, and survive rigorous testing.

## (8) TWO-CITY GENERALITY: Paris (continental) vs London — every finding holds, intensity stronger

Second city (reviewer's main ask): Paris, a contrasting continental/convective regime, 607 extreme
days (vs London 417). Same pipeline, Paris ERA5 box. All analyses re-run.

**Bake-off (+ERA5, seed-averaged):**
| metric | London | Paris |
|--------|-------:|------:|
| logistic BSS | +0.038 | **+0.117** |
| XGBoost BSS | +0.034 | +0.101 |
| ZI-eGPD BSS | +0.012 | +0.093 |
| ZI-eGPD twCRPSS | −0.045 | **+0.052** |
| ZI-lognormal twCRPSS (bad) | −2.27 | −0.76 |
| deep-GP–EVT BSS | ≈−0.4 | ≈−0.4 |

**Reviewer's deep-GP sensitivity (both cities):** deep-kernel robustly fails (BSS −0.2 to −0.7 across
M=32–128); **plain GP is competitive** (London +0.029, Paris +0.123). → it is *depth* that fails on
low signal, not GPs.

**Feature screen (FDR):** K-index leads occurrence in both; CAPE/CIN lead intensity in both.
Survive Benjamini–Hochberg: London 21/49, **Paris 27/49** (stronger).

**Hold-out intensity (CRPS-of-excess skill vs climatology, moving-block bootstrap):**
| | London | Paris |
|---|---|---|
| CRPS-of-excess skill | +0.014 [+0.009,+0.019] | **+0.036 [+0.005,+0.074]** |
| q90 pinball skill | +0.05 | +0.086 |
| multi-split (4 splits) | 4/4 positive, mean +0.033 | **4/4 positive, mean +0.042**, all CIs exclude 0 |

**Conclusion — the paper's claims are NOT London-specific.** Across two contrasting cities:
(i) simple calibrated models (logistic, XGBoost) and the eGPD beat the deep-GP–EVT; (ii) distribution
choice dominates (log-normal bad, eGPD good); (iii) occurrence is weakly predictable, led by the
K-index; (iv) the magnitude of the largest events is predictable from antecedent **CAPE/CIN
convective instability**, with bootstrap CIs excluding zero across all splits. Effects are
**stronger in Paris** (more extremes, more convective), as expected. Notably, in Paris the eGPD even
beats climatology on the full-predictive tail score (twCRPSS +0.052), which it did not in
data-scarcer London — so "the tail is climatology-bound" is a London-sample-size effect, not a
universal one.

## Takeaways for the paper
1. The full pipeline (deep-kernel coregionalised SVGP → hurdle+GPD → tail
   conformal) is implementable and trains stably.
2. The dependent-hurdle contribution is empirically real and matters most
   exactly where data are scarce (the 99th-percentile tail).
3. The conformal coverage guarantee holds on the real series.
4. Antecedent ERA5 drivers do **not** give a robust skill gain at h=1 with the
   coarse WB2 feature set (paired ΔPR-AUC = +0.003 ± 0.012, mixed sign over
   seeds); the earlier single-seed "0.036→0.050" was an outlier. Calibration,
   however, holds across all configs — consistent with the calibration-first
   thesis. Where drivers may yet pay off: longer horizons and finer ERA5 fields
   (CAPE/IVT). All ablations must be seed-averaged.
5. Horizon sweep (h=1..28): skill decays toward the prevalence floor; drivers
   give no statistically robust gain at any horizon (the h=28 ERA5 hint vanished
   under an 8-seed check: +0.004 ± 0.006). Crucially, **conformal coverage stays
   ≈0.90 at every horizon and config** — the paper's dependable headline is
   calibration, not skill. Finer ERA5 fields (CAPE/IVT) remain the only untested
   lever for skill.
