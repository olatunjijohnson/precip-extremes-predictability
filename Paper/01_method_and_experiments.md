# Method and Experiment Plan

**Headline:** *Tail-calibrated conformal deep Gaussian process for rare-event precipitation forecasting.*
See `00_improvement_roadmap.md` for the high-level direction. This document is the technical plan
and the literature positioning.

---

## 1. Literature check — what already exists, and our gap

The intersection of EVT, conformal prediction, and deep probabilistic models has become **active in
2025–2026**, so the contribution must be scoped precisely. Closest prior work:

| Work | What it does | What it does NOT do (our gap) |
|------|--------------|-------------------------------|
| **Extreme Conformal Prediction** (Pasche, Lam & Engelke, 2025; arXiv 2505.08578; to appear *Extremes*) | EVT + conformal to build reliable **regression intervals at very high confidence**; weighted variant for nonstationarity; flood-risk demo | No Gaussian process / deep model; no generative predictive distribution; no *exceedance-probability* forecasting; no explicit temporal-dependence treatment |
| **DeepExtrema** (Galib et al., IJCAI 2022; arXiv 2205.02441) | Deep network forecasting **block maxima** via GEV | No GP; no conformal; GEV/block-maxima not POT/GPD; no coverage guarantee |
| **NN for GEV/GPD parameters** (Rai et al., *Environmetrics*, 2024) | Fast NN estimation of EVT parameters | Not forecasting; no conformal; no GP; no calibration guarantee |
| **Adaptive Conformal Inference** (Gibbs & Candès, NeurIPS 2021; arXiv 2106.00170) | Conformal under distribution shift / time series (the building block we use) | Generic, not tail/extreme-specific; not a forecasting model |
| **Valid error bars for neural weather models** (arXiv 2406.14483, 2024) | Conformal wrappers on weather models | Not tail-specialised; no EVT; no GP |
| **INLA goes extreme** (Opitz et al., arXiv 1802.01085) | Bayesian spatio-temporal tail regression with GP priors | Not forecasting; no conformal; no deep model |

**The white space (our contribution).** No existing paper provides a **generative deep-GP model with
a Generalised Pareto output layer, wrapped in a time-series-adaptive conformal layer that targets
tail / exceedance coverage**, evaluated with proper weighted scoring rules on a climate-forecasting
task with antecedent physical drivers. The closest (Extreme Conformal Prediction) produces
*intervals* from a black-box quantile model under (weighted) exchangeability — we produce a *full,
generative, calibrated predictive tail* under temporal dependence. That is a clean, defensible
delta.

**Three precise differentiators to state in the paper:**
1. **Generative** DGP+GPD predictive distribution (occurrence + intensity hurdle), not a black-box
   interval wrapper.
2. **Temporal dependence handled natively** via adaptive/blocked conformal, not iid/exchangeable.
3. **Calibration-first** evaluation centred on tail reliability + threshold-weighted CRPS, framed by
   the *forecaster's dilemma* (Lerch et al., 2017, arXiv 1512.09244).

---

## 2. Problem setup and notation

- Daily standardised precipitation index $\mathcal{I}_t \ge 0$ for London; extreme if
  $\mathcal{I}_t > u$ (use $u=1$, the existing threshold; sensitivity to $u$ as an experiment).
- Covariates $\mathbf{x}_t$ = **strictly antecedent** features known at time $t$:
  autoregressive lags + rolling stats (existing) **plus** lagged ERA5 drivers (Section 7) and
  teleconnection indices.
- Forecast target: the predictive distribution of $\mathcal{I}_{t+h}$ for horizons $h\in\{1,3,7\}$
  (and a sub-seasonal stretch, e.g. $h=14,28$).
- Two linked quantities: **occurrence** $p_{t+h}=P(\mathcal{I}_{t+h}>u\mid \mathbf{x}_t)$ and
  **intensity** $\mathcal{I}_{t+h}\mid \mathcal{I}_{t+h}>u$.

---

## 3. Model — deep GP with a hurdle + GPD output layer

A standard GP with Gaussian likelihood **cannot represent heavy-tailed, skewed, zero-inflated**
responses (confirmed in the DGP literature). We therefore use depth for flexible warping and an
EVT likelihood for the tail.

**Latent structure.** A 2-layer deep GP (or deep-kernel GP — see 3.1) maps
$\mathbf{x}_t \to \mathbf{f}_t$, a low-dimensional latent vector. From $\mathbf{f}_t$ three heads:

1. **Occurrence head** — $p_{t+h} = \sigma(g_\pi(\mathbf{f}_t))$ (Bernoulli, the hurdle gate).
2. **Bulk head** — distribution of the non-extreme part (e.g. log-normal / gamma for
   $0<\mathcal{I}\le u$).
3. **Tail head — Generalised Pareto** for exceedances $(\mathcal{I}-u)\mid \mathcal{I}>u \sim
   \mathrm{GPD}(\sigma_{t+h}, \xi)$, with **non-stationary scale** $\sigma_{t+h}=\exp(g_\sigma(\mathbf{f}_t))$
   and shape $\xi$ (start constant; allow covariate-dependent $\xi_{t+h}$ as an extension).

**Predictive distribution** is the hurdle mixture:
$$
F(\mathcal{I}\mid\mathbf{x}_t)=
\begin{cases}
1-p_{t+h} & \text{(point mass / bulk below } u)\\
p_{t+h}\,\mathrm{GPD}(\mathcal{I}-u;\sigma_{t+h},\xi) & \mathcal{I}>u.
\end{cases}
$$
This yields, for free, calibrated **exceedance probabilities** and **extreme quantiles**.

### 3.1 Depth as an ablation (answering "why not a plain GP?")
Compare, as a deliberate ablation ladder:
- (i) single-layer GP + GPD output (no depth),
- (ii) **deep-kernel GP** (NN feature extractor + GP layer; most tractable — likely the workhorse),
- (iii) **2-layer deep GP** (doubly-stochastic variational inference).
Report whether depth improves *tail* calibration/sharpness specifically. Honesty: full DGP is
data-hungry and inference is fragile with few exceedances — DKL is the safe default; depth is a
tested ingredient, not a fragile centrepiece.

---

### 3.2 Extension — *dependent* hurdle: borrowing strength between occurrence and intensity

The classical hurdle model assumes occurrence ⟂ intensity given covariates. That is often false:
the same synoptic conditions that make an extreme *more likely* also make it *more intense*, so the
two parts should be **positively correlated**. Modelling this lets the abundant occurrence data
inform the data-starved tail — exactly the "borrowing strength" we want given so few exceedances.
This is well-precedented (correlated two-part / shared-random-effect models, Olsen & Schafer 2001;
copula two-part models; sample-selection / Heckman 1979) but **not** done in a deep-GP–EVT
forecasting setting — so it is a genuine extension, not a re-tread.

**How to do it in our GP framework (natural fit):** make the occurrence latent $g_\pi$ and the
GPD-scale latent $g_\sigma$ **correlated outputs of a multi-output (coregionalised) GP** rather than
two independent GPs:
$$\mathrm{Cov}\big(g_a(\mathbf{x}_t),\,g_b(\mathbf{x}_{t'})\big)=B_{ab}\,k(\mathbf{x}_t,\mathbf{x}_{t'}),
\qquad a,b\in\{\pi,\sigma\},$$
with a learned $2\times2$ PSD coregionalisation matrix $B$ (intrinsic coregionalisation model /
LMC; Bonilla et al. 2008; Álvarez et al. 2012). The off-diagonal $B_{\pi\sigma}$ is the
**interpretable correlation parameter** = the strength of borrowing. (In the deep variant, both
heads already share the latent $\mathbf{f}_t$; coregionalisation makes the coupling explicit and
estimable.) Inference is unchanged in form — joint likelihood = Bernoulli(occurrence) ×
GPD(intensity on exceedance days) — with the correlated latent tying them.

**Caveats (state honestly):** intensity is observed only on extreme days, so $B_{\pi\sigma}$ is
weakly identified with few exceedances — use priors/regularisation, fit constant shape $\xi$ first,
and **validate on simulation with known true correlation**. Test value via ablation: independent
hurdle vs. coregionalised hurdle (add to the ablations in §9). Alternative routes if coregionalised
GP is unstable: a Gaussian-copula coupling of the two margins, or correlated random effects.

## 4. Inference

- Variational inference with inducing points (sparse GP) — fixes the $O(n^3)$ problem properly
  (the old code's crude 400-point subsample is replaced by principled inducing points).
- Doubly-stochastic VI for the deep variant (Salimbeni & Deisenroth, 2017).
- Tooling: **GPyTorch** (preferred) or **GPflow**; EVT likelihood as a custom likelihood module.
- Learn kernel hyperparameters (the old code froze them — a key reason the GP failed).

---

## 5. Tail-conformal calibration layer (the methodological core)

The DGP+GPD gives a model-based predictive tail; conformal makes it **distribution-free calibrated**.

- **Score:** nonconformity on exceedance probability / extreme-quantile residuals, not just point
  residuals — so calibration is targeted at the tail (cf. Extreme Conformal Prediction, but inside a
  generative forecaster).
- **Temporal dependence:** use **adaptive conformal inference** (Gibbs–Candès) / blocked conformal
  on a rolling calibration window, instead of assuming exchangeability. Walk-forward calibration
  matches the forecasting setting.
- **Guarantee to claim:** long-run / rolling marginal coverage of the exceedance event and of the
  extreme-quantile interval, under temporal dependence — empirically with approximate conditional
  (tail) coverage.

---

## 6. Baselines (must beat or match on calibration, be honest where it ties)

1. **Climatology** (seasonal base rate) — the calibration sanity floor; we must be *sharper*.
2. **Persistence.**
3. **Logistic regression** (the current surprise winner) — occurrence only.
4. **XGBoost hurdle / quantile** (the current ML model).
5. **Single-layer GP + Gaussian likelihood** (the current failing GP — shows the EVT layer matters).
6. **Stationary POT/GPD (EVT only)** — no ML covariates; shows covariates matter.
7. **Conformalised quantile regression** (Romano et al., 2019) — standard conformal, to show tail
   under-coverage that our tail-conformal fixes.

---

## 7. Data

- **Index:** existing London standardised index, 1989–2018, 95th & 99th percentile (already cleaned).
- **ERA5 drivers (decided: full ERA5, strictly antecedent):** MSLP, geopotential Z500/Z850,
  specific & total-column water vapour, IVT (integrated vapour transport), CAPE, 2 m temperature,
  10 m winds, over a London-centred box. Lag by $\ge 1$ day; **no concurrent/future fields**
  (ERA5 is reanalysis → concurrent use is circular). Details/recipe to go in `02_era5_data.md`.
- **Teleconnection indices:** NAO, AO, ENSO (small public series) — cheap, physically motivated.
- **Split:** train 1989–2008, test 2009–2018; walk-forward CV retained.

---

## 8. Evaluation — calibration-first suite

**Primary (calibration / tail):**
- Reliability diagrams overall **and restricted to the tail**; exceedance-probability calibration.
- **Threshold-weighted CRPS** (Gneiting & Ranjan, 2011) emphasising values $>u$.
- Prediction-interval / extreme-quantile **coverage** vs nominal (and conditional/tail coverage).
- Brier score **reliability–resolution decomposition**; Brier Skill Score & **CRPSS** vs climatology.

**Secondary (discrimination — "remains competitive"):** PR-AUC, ROC-AUC, F1.

**Rigour:** bootstrap CIs on all metrics; significance tests between models; report under the
*forecaster's dilemma* framing to justify weighted scoring.

---

## 9. Experiments / ablations

1. **Main comparison** — all models, all horizons, both percentiles, full metric suite.
2. **EVT-layer ablation** — GPD output vs Gaussian likelihood (isolates the tail contribution).
3. **Depth ablation** — single GP vs deep-kernel GP vs 2-layer DGP (Section 3.1).
4. **Conformal ablation** — no conformal vs standard (exchangeable) conformal vs tail/adaptive
   conformal (isolates the calibration contribution and shows tail under-coverage of standard CP).
5. **Covariate ablation** — index-only vs +teleconnections vs +full ERA5 (quantifies driver value;
   also the "is ERA5 a cheat?" check — gains must come from *antecedent* info only).
6. **Dependence ablation (the §3.2 extension)** — independent hurdle vs coregionalised (dependent)
   hurdle; report the learned correlation $B_{\pi\sigma}$ and whether borrowing strength improves
   *tail* calibration/sharpness, especially at the 99th threshold where exceedances are scarcest.
   Include a simulation with known true correlation to confirm recovery.
7. **Horizon degradation** + sub-seasonal stretch ($h=14,28$).
8. **Threshold sensitivity** — $u$ at 90/95/99th.

---

## 10. Tooling & reproducibility
- Python: GPyTorch/GPflow, PyTorch, `scipy.stats.genpareto`, `properscoring` / custom twCRPS,
  `xgboost`, `scikit-learn`.
- ERA5 via Copernicus CDS API; xarray/netCDF processing.
- Fixed seeds; config-driven; everything walk-forward to avoid leakage.

---

## 11. Risks & mitigations
- **DGP/GPD inference instability with few exceedances** → DKL as default; constant $\xi$ first;
  strong priors/regularisation; pool 95th-threshold exceedances to stabilise tail estimation.
- **ERA5 size/compute** → restrict to a London box and key levels; precompute daily features.
- **Crowded literature (2025–26)** → keep the three differentiators (Section 1) front and centre;
  do the EVT-layer + tail-conformal ablations that competitors lack.
- **Skill may stay modest** → that's expected; the paper's value is *calibration + guarantees*,
  not raw skill (the framing protects against it).

---

## 12. Key references
All entries are maintained in **`references.bib`** (cite keys in parentheses):
Pasche, Lam & Engelke 2025 (`pasche2025extreme`); Gibbs & Candès 2021 (`gibbs2021adaptive`);
Romano, Patterson & Candès 2019 (`romano2019cqr`); Damianou & Lawrence 2013 (`damianou2013deep`);
Salimbeni & Deisenroth 2017 (`salimbeni2017doubly`); Wilson et al. 2016 (`wilson2016deepkernel`);
Rasmussen & Williams 2006 (`rasmussen2006gpml`); Coles 2001 (`coles2001evt`);
Gneiting & Ranjan 2011 (`gneiting2011threshold`); Gneiting & Raftery 2007 (`gneiting2007scoring`);
Lerch et al. 2017 (`lerch2017dilemma`); Galib et al. 2022 (`galib2022deepextrema`);
Olsen & Schafer 2001 (`olsen2001twopart`); Bonilla et al. 2008 (`bonilla2008multitask`);
Álvarez et al. 2012 (`alvarez2012kernels`); Heckman 1979 (`heckman1979selection`).
