# Payment Failure Prediction & Smart-Retry A/B Test

End-to-end analytics project on a payment-gateway dataset of **200,000 transactions**.
It answers two different questions with two different methods:

1. **Did the change we shipped work?** — a randomised A/B test of a new retry-routing policy, analysed with hypothesis tests, a power calculation and a multiple-comparisons correction.
2. **Which payment attempt will fail next?** — a classification model comparing Logistic Regression, Decision Tree, Random Forest, XGBoost and KNN.

The dataset contains deliberate, documented flaws — target leakage, multicollinearity, a high-cardinality categorical, sensor-glitch outliers, a U-shaped effect, an interaction, class imbalance and a right-skewed regression target — so that the analysis has something real to find.

---

## Two versions of the notebook

| Notebook | Size | Runtime | Use it for |
|---|---|---|---|
| **`payment_failure_simple.ipynb`** | 84 cells | ~7 min | **Start here.** The main portfolio piece — the full story with plain, step-by-step code |
| `payment_failure_prediction.ipynb` | 146 cells | ~20 min | The extended version — hyperparameter tuning, calibration, permutation importance, a regression stretch goal, and deeper statistical commentary |

Both use the same dataset and reach the same conclusions. The simple one is what you should
walk an interviewer through; the extended one is where to look up a deeper answer if they push.

**What the simple version keeps:** cleaning with the reasoning behind each fix, EDA, leakage detection, the full A/B test, all five models, the feature-engineering fix, cost-based threshold selection, and the recommendation.

**What only the extended version has:** hyperparameter tuning, model calibration, permutation importance, a per-model cost comparison, ablation studies, VIF, Cramér's V, Mann-Whitney, and a linear-regression stretch goal on `revenue_loss_inr`.

---

## The headline results

### A/B test — `smart_retry` routing

| | Control | Smart retry |
|---|---:|---:|
| Transactions | 99,772 | 99,578 |
| Failure rate | 16.53% | **15.49%** |

- **Absolute reduction: 1.03 pp**, 95% CI **[0.71, 1.35] pp**, z = 6.29, p = 3.3×10⁻¹⁰
- **Relative reduction: 6.25%** of the control failure rate
- **Annualised revenue recovered at full rollout: ~₹1.88M** (CI ₹1.30M – ₹2.47M)
- **Guardrail (latency): PASS** — +6.1 ms, statistically significant (p = 6×10⁻⁵⁷) but Cohen's d ≈ 0.07, i.e. practically negligible
- **Segments:** the effect is concentrated in **UPI** (1.62 pp, survives Bonferroni). Debit Card looks significant at α = 0.05 but **does not survive correction for 6 tests** and is reported as a hypothesis, not a finding.

### Model comparison (held-out test set, 39,870 rows)

| Model | ROC-AUC | PR-AUC | Train−test gap |
|---|---:|---:|---:|
| XGBoost (tuned) | **0.9335** | **0.7897** | 0.012 |
| XGBoost (default-ish) | 0.9322 | 0.7848 | 0.033 |
| **Logistic regression (engineered)** | **0.9285** | **0.7737** | **0.002** |
| Random Forest (tuned) | 0.9174 | 0.7323 | 0.033 |
| Random Forest (300 trees) | 0.9129 | 0.7179 | 0.044 |
| Decision Tree (depth 8) | 0.8810 | 0.6598 | 0.009 |
| KNN (k = 35, scaled) | 0.8444 | 0.5482 | 0.045 |
| Logistic regression (raw features) | 0.8196 | 0.5329 | 0.001 |

*(Numbers above are from the extended notebook, which includes a tuning round. The simple notebook skips tuning, so its Random Forest is 0.9127 and its XGBoost 0.9324 — the ranking and the story are identical.)*

**Cost comparison at each model's own optimal threshold** (₹450 per false negative, ₹30 per false positive, on the 39,870-row test set):

| Model | Optimal threshold | Recall | Test cost | vs best |
|---|---:|---:|---:|---:|
| XGBoost (tuned) | 0.20 | 0.957 | ₹459,720 | — |
| **Logistic (engineered)** | **0.28** | **0.934** | **₹469,380** | **+2.1%** |
| Random Forest (tuned) | 0.25 | 0.941 | ₹497,010 | +8.1% |
| Logistic (raw features) | 0.23 | 0.935 | ₹803,190 | +74.7% |

**The finding that matters:** the raw logistic regression loses to the trees by **0.09 AUC**. Adding four engineered terms — a quadratic, an interaction and two logs, all functions of columns the model already had — closes **97%** of it. The trees' advantage was never model capacity; it was that they find non-linearity and interactions without being told, and the gap is a measurement of how much structure has not yet been encoded.

**Shipped model:** logistic regression with the engineered terms, at a **cost-optimal threshold of 0.28** rather than 0.5 — chosen because a false negative (a lost sale, ~₹450) costs roughly 15× a false positive (an unnecessary reroute, ~₹30). Moving the threshold off the 0.5 default is worth **18.7% of expected cost** and costs nothing. XGBoost is genuinely more accurate, but only by **2.1% of expected cost**, which is what has to be weighed against microsecond inference and an auditable coefficient table; it is kept as a monitored challenger, and the write-up states the condition under which the decision flips.

---

## Repository contents

| File | What it is |
|---|---|
| `payment_failure_simple.ipynb` | **The main notebook** — executed end to end, 7 charts |
| `payment_failure_prediction.ipynb` | The extended version, executed end to end, 18 charts |
| `payments_ab_failure_200k.csv` | The dataset (201,200 rows × 35 columns, ~40 MB) |
| `generate_dataset.py` | The generator, with every embedded property documented |
| `EVALUATION_AND_INTERVIEW_PREP.md` | Hiring-manager scoring against the rubric, design-decision rationale, and 34 interview questions with answers |
| `README.md` | This file |

## How to run

```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib jupyter
jupyter notebook payment_failure_simple.ipynb
```

The simple notebook runs top to bottom in about 7 minutes on 2 CPU cores; the extended one takes 15–20. To regenerate the dataset (the seed is fixed, so it reproduces exactly):

```bash
python generate_dataset.py
```

In Google Colab, upload `payments_ab_failure_200k.csv` alongside the notebook and run all cells — `xgboost` is preinstalled.

---

## What the simple notebook covers

1. **Cleaning** — duplicates, a constant column, inconsistent casing, impossible values, missing-value strategy
2. **Exploring** — class balance, the two outlier populations, correlation, the U-shaped effect, the latency × load interaction, categories
3. **Leakage check** — three post-outcome columns found and dropped
4. **A/B test** — split check, z-test with a confidence interval, power, business impact, latency guardrail, segments with Bonferroni
5. **Model setup** — train/test split, two preprocessing recipes and why they differ
6. **Training** — five models, the overfitting demo, the KNN scaling demo, then the feature-engineering fix
7. **Evaluation** — ROC and PR curves, cost-based threshold, confusion matrix, coefficients
8. **Recommendation** — what to ship, what to check before trusting it, what it does not prove

---

## What the extended notebook adds

**1–3. Business framing and data preparation**
Cost of a false negative vs a false positive established up front and used later to pick the threshold. Duplicate webhook rows, a constant column, inconsistent categorical casing, whitespace-padded IDs, `"unknown"` masquerading as a category, sentinel ages (0 / 999) and wrongly-signed refund rows are each handled — with the reason for the *specific* fix, since "convert to NaN and impute" and "drop the row" solve different problems. Row-local cleaning happens before the split; anything that learns a statistic happens inside the pipeline, after it.

**4. Exploratory analysis**
Class balance and why accuracy is rejected as a metric. Skew and outlier diagnosis: the IQR fence flags 6% of latency readings, but they are **two populations** — 2.05% are genuinely slow and fail at **33%** (double the base rate, real signal), while 3.99% sit above 5,000 ms and fail at **16.7%**, indistinguishable from the 16.0% base rate (instrumentation glitches). A 99th-percentile winsorisation would have missed the glitches entirely, since the contamination is 4%; the notebook uses a domain cap instead and explains why. Correlation matrix, the r = 0.95 pair, and VIF computed from first principles. **Binned-mean plots that reveal a U-shaped effect a Pearson r of 0.21 badly understates** — failure risk bottoms out at 5.8% near 95 seconds and exceeds 85% at the extremes, and destroying that one column costs 0.19 of test AUC — and a 2D heatmap revealing a latency × load interaction that **reverses sign** — slower is safer at low load (−11.3 pp), far more dangerous at high load (+16.6 pp), which is why latency on its own has a single-feature AUC of 0.52. Chi-square tests with Cramér's V alongside, because at n = 200,000 almost everything is significant and only the effect size tells you whether it matters.

**5. Leakage audit**
Three post-outcome columns identified by asking *when is this value written*, quantified by single-feature AUC (0.99, 0.92, 0.70 versus 0.64 for the best legitimate feature), and dropped. Leaving one in would have produced a ~0.97 AUC model that predicts "success" for everything in production.

**6. A/B testing**
SRM check before anything else. Covariate balance with standardised differences — **excluding the post-treatment variable**, with the reason. Power and MDE computed and used to justify the sample size. Two-proportion z-test with pooled SE for the statistic and unpooled for the interval, plus the equivalent chi-square. Business impact reported as an interval. Guardrail analysed with both Welch's t-test and Mann-Whitney, then dismissed on effect size. Segment analysis with Bonferroni. Peeking, unit of randomisation, SUTVA and external validity all addressed.

**7–9. Modelling**
Two preprocessing pipelines, not one, with a table explaining every difference by model mechanism. Logistic baseline, Decision Tree (plus an unregularised one shown overfitting at train AUC 1.000 / test 0.724), Random Forest, XGBoost, and KNN with k chosen by CV, an ablation of the 45-level categorical, and a measurement of what forgetting to scale costs (0.148 AUC). Then feature engineering, with an ablation showing which term did the work. Tuning by randomised search on a subsample, cross-validated on train only.

**10. Evaluation**
ROC and PR curves side by side. Threshold selection from an explicit cost function, with F1's implicit equal-cost assumption called out as wrong here. A model-vs-model comparison at each model's own cost-optimal threshold — the decision denominated in rupees rather than AUC. Calibration curves, and why class weighting distorts them.

**11. Regression stretch**
Skew and kurtosis checked before modelling. Heteroscedasticity *measured* rather than eyeballed. Log transform tested, Duan's smearing correction applied — and the honest finding that the log model halves the median error while worsening RMSE, because `exp()` amplifies tail errors.

**12. Recommendation**
A decision, the trade-off it accepts, a pre-launch checklist, a post-launch monitoring plan including the feedback-loop risk, and a limitations section.

---

## Data dictionary

**Identifiers and time**

| Column | Type | Description |
|---|---|---|
| `txn_id` | str | Transaction key. ~1,200 rows are duplicated (double-fired webhooks) |
| `txn_timestamp` | str | Attempt timestamp, Jan–Jun 2026 |
| `txn_hour`, `is_weekend` | int | Derived time fields |
| `merchant_id` | str | 1,500 merchants; ~2% have trailing whitespace |
| `data_version` | str | Constant `"v2.1"` — zero variance, should be dropped |

**Categorical**

| Column | Levels | Notes |
|---|---|---|
| `payment_method` | 6 | UPI, Credit Card, Debit Card, Net Banking, Wallet, EMI. Real baseline hazard differences |
| `merchant_category` | 10 (+`"unknown"`) | ~1.5% are the string `"unknown"` — missing wearing a category's clothes |
| `card_network` | 5 | **Inconsistent casing/padding** — appears as 20 distinct strings before cleaning |
| `issuer_bank_code` | 45 | **High cardinality, mostly noise.** ~5 banks genuinely differ; Cramér's V ≈ 0.05 |
| `device_os`, `region`, `customer_tier` | 4 / 6 / 3 | Low-signal context |
| `experiment_group` | 2 | `control` vs `smart_retry`, randomised ~50/50 |

**Numeric features**

| Column | Notes |
|---|---|
| `txn_amount_inr` | Right-skewed; ~650 rows are **negative** (refunds that leaked into the feed) |
| `session_duration_sec` | **U-shaped effect** on failure, optimum ≈ 90s. Pearson r with the target ≈ 0.01 |
| `network_latency_ms` | **~4% sensor-glitch spikes** (6,000–25,000 ms) that carry no signal. Interacts with load |
| `gateway_load_pct` | Interaction partner for latency |
| `merchant_tenure_months` | **Log-saturating** protective effect |
| `merchant_avg_ticket_size_30d` | Correlated **r ≈ 0.95** with the next column |
| `merchant_gmv_per_txn_30d` | The same quantity measured a second way — the multicollinearity trap |
| `merchant_success_rate_30d` | Rolling window; worth confirming it is *trailing* and not recomputed |
| `merchant_risk_score` | ~3% missing |
| `device_trust_score` | ~7% missing |
| `customer_age` | ~10% missing, plus ~1,100 sentinel values of `0` and `999` |
| `prior_failed_attempts_24h`, `retry_attempt_number` | The strongest legitimate signals |
| `is_international`, `kyc_verified`, `card_bin_country_match` | Binary flags |

**Post-outcome columns — must be excluded (target leakage)**

| Column | AUC alone | Why it leaks |
|---|---:|---|
| `gateway_error_code` | 0.99 | Emitted *because* the payment failed; null on 98% of successes |
| `retry_within_10min_flag` | 0.92 | The customer retries *after* seeing the failure |
| `support_ticket_raised` | 0.70 | Raised hours to days later |

**Targets**

| Column | Notes |
|---|---|
| `payment_failed` | Binary, ~16% positive |
| `revenue_loss_inr` | Continuous, `> 0` only for failed attempts. Skew ≈ 6.7, kurtosis ≈ 117 — the regression stretch goal |

---

## Techniques demonstrated

**Data preparation** — deduplication, constant-column removal, categorical normalisation, sentinel-value handling, impossible-value removal, missingness testing, pipeline-based imputation, train-only winsorisation

**Statistics** — chi-square (goodness of fit and independence), Cramér's V, two-proportion z-test, Welch's t-test, Mann-Whitney U, Cohen's d, confidence intervals, VIF, skewness/kurtosis, heteroscedasticity testing, Duan's smearing estimator, Bonferroni correction

**Experimentation** — SRM detection, covariate balance, power and MDE calculation, guardrail metrics, segment analysis, peeking, unit-of-randomisation and SUTVA considerations

**Machine learning** — logistic regression, decision trees, random forests, XGBoost, KNN, linear/ridge regression, random-forest regression; `Pipeline` / `ColumnTransformer`, stratified splitting, cross-validation, randomised hyperparameter search, class weighting, threshold optimisation from a cost function, calibration, permutation importance, and target-leakage detection
