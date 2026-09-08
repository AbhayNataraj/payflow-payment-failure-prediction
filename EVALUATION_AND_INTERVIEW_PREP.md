# Self-Evaluation & Interview Preparation
### Project: Payment Failure Prediction + Smart-Retry A/B Test (PayFlow)

This document has three parts:

1. **Part A — Hiring-manager evaluation.** The project scored against the rubric, with an honest hire / no-hire call and the specific things that cost marks.
2. **Part B — Design decisions.** Every significant choice: what was used, why, what the alternatives were, and why this one won.
3. **Part C — Interview questions.** Questions this project invites, plus general ML/stats questions, each with the answer you should be able to give.

---
---

# PART A — Hiring-Manager Evaluation

*Written as if I received this as a take-home submission and had 30 minutes to review it.*

## A.1 Score

| Area | Weight | Score | Notes |
|---|---:|---:|---|
| **EDA quality** — missingness, correlation, class balance, outliers surfaced | 15 | **14** | Missingness is not just reported but *tested for informativeness* before choosing an imputation strategy. Outliers are not just flagged but **split into two populations** — 2% genuinely slow (33% failure rate, real signal) and 4% instrumentation glitches (16.7%, base rate) — and the candidate then notices that a 99th-percentile cap would sit *inside* the 4% contamination and switches to a domain cap. That is a level of care I rarely see. The binned-mean plots are the right instinct. Lost a point for no bivariate exploration of the categorical × categorical space. |
| **Leakage detection** — the hard gate | 20 | **20** | All three leaky columns identified, quantified by single-feature AUC, reasoned about by *when the value is written*, and dropped. Bonus: `merchant_success_rate_30d` flagged as needing a conversation about whether its window is trailing. This is exactly the behaviour I want. |
| **Logistic regression rigour** — scaling, encoding, imputation, no train/test leakage | 15 | **15** | Percentile caps learned on train only; latency capped by a domain constant instead, with the reason. Imputer and scaler inside the pipeline, fitted after the split. Two separate preprocessors with the difference justified per model family. Class weighting explained rather than pasted in. |
| **Diagnosing & fixing non-linearity / interaction** — the differentiator | 20 | **20** | The candidate found the U-shape via binned means despite a Pearson r of 0.21 that would have ranked the feature as middling; found an interaction that **reverses sign** with load, and connected it to why latency alone has a single-feature AUC of 0.52; engineered exactly two terms to fix both, moving logistic regression from 0.8196 to 0.9285 — closing **97%** of the gap to XGBoost. The ablation showing which term did the work is above expectation. |
| **Tree models used correctly** — minimal preprocessing, tuning, overfitting checks | 15 | **14** | No scaling for trees, with the reason stated. Unregularised tree shown overfitting (train AUC 1.000 / test 0.724) as a deliberate contrast. `scale_pos_weight` derived from the data. Randomised rather than exhaustive search, with the reason. Half a point off for tuning on a 60k subsample and for not using XGBoost's native NaN handling in practice after mentioning it. |
| **KNN done correctly** — scaling, k selection, dimensionality comment | 5 | **5** | k chosen by CV, not guessed. Quantified the cost of forgetting to scale (+0.148 AUC). Ran the high-cardinality ablation, got a null result, and **reported the null result honestly with a corrected explanation** rather than quietly deleting the experiment. That is worth more than the effect would have been. |
| **Evaluation & metric choice appropriate to imbalance** | 5 | **5** | Accuracy explicitly rejected with the 84% baseline stated. PR-AUC led over ROC-AUC with a reason. Threshold chosen from a cost function, not left at 0.5, and F1's implicit assumption called out as wrong for this problem. Calibration checked, with the `class_weight` distortion explained. |
| **Communication & final recommendation** | 5 | **5** | The recommendation names a model, a threshold, the trade-off it accepts, and the condition under which the answer would flip. Limitations section is unusually candid. |
| **Subtotal (original rubric)** | **100** | **98** | |

### Supplementary areas (this brief additionally required A/B testing, statistics and data preparation)

| Area | Weight | Score | Notes |
|---|---:|---:|---|
| **Data preparation** | 25 | **24** | Duplicates, a constant column, inconsistent casing, whitespace, `"unknown"`-as-a-category, sentinel values and wrongly-signed amounts all handled — and each with the *reason* for the specific fix chosen (impute vs drop, and why they differ). Correctly separated row-local cleaning (safe before the split) from learned transformations (must be after). |
| **A/B testing** | 25 | **24** | SRM check first. Covariate balance with standardised differences. **Correctly excluded a post-treatment variable from the balance check** — this is the detail that separates people who have run experiments from people who have read about them. Power/MDE computed and used to justify the sample size. Two-proportion z-test with a CI, chi-square shown to be the same test. Guardrail analysed with both a parametric and a non-parametric test. Segment analysis with Bonferroni, and the borderline segment correctly *not* claimed. Peeking, unit of randomisation, SUTVA and novelty all raised. One point off for not showing a sequential-testing alternative in code. |
| **Statistics beyond the tests** | 15 | **15** | VIF computed from first principles rather than imported. Cramér's V used to separate significance from effect size at n = 200k. Skewness/kurtosis checked before choosing a regression target transform. Heteroscedasticity *measured* (corr between |residual| and fitted) rather than eyeballed. Duan's smearing correction applied and — more impressively — correctly explained when it *hurts*. |
| **Regression (stretch)** | 10 | **9** | Distribution checked before modelling. Log transform tested rather than assumed. The honest finding that the log model wins on median error but loses on RMSE, with the reason (`exp()` amplifies tail errors), is better analysis than a clean win would have been. |
| **Business framing** | 10 | **10** | Cost of a false negative vs a false positive stated up front, in rupees, and then actually used to pick the threshold and to compare models. Experiment result converted to annualised revenue **with a confidence interval**, not a point estimate. |
| **Code quality & reproducibility** | 15 | **13** | Clean sklearn `Pipeline` / `ColumnTransformer` throughout, seeded, no copy-paste blocks, a single `evaluate()` helper. Two points off: no `requirements.txt` pinning, and the notebook takes ~20 minutes to run end-to-end with no cached-results path. |
| **Subtotal (supplementary)** | **100** | **95** | |

### **Overall: 96 / 100**  *(rubric 98, supplementary 95)*

## A.2 Would I hire this candidate?

**Yes — strong hire** for a Data Analyst, Analytics Engineer, or junior/mid Data Scientist role. If the loop went well I would push for the upper end of the band.

**What convinced me, in order:**

1. **They caught the leakage and reasoned about it correctly.** This is the single trait that separates people who can be trusted with a production model from people who cannot. Plenty of candidates get a 0.97 AUC and celebrate. This one asked *when is this column written* and threw away ~0.04 AUC to be right. That is the whole job.

2. **They asked "why" when the numbers were disappointing.** Logistic regression lost round one by 0.09 AUC. The common response is "trees are better on tabular data" and move on. This candidate went back to the EDA, found the mechanism, fixed it with two terms, and closed almost the entire gap. That is the difference between someone who runs models and someone who understands them.

3. **They reported a null result.** The KNN high-cardinality ablation did not show what they expected. Instead of deleting it, they kept it, corrected their explanation, and articulated *why* the textbook effect was muted. Candidates who only show their wins are candidates whose numbers you cannot trust.

4. **The A/B section is done by someone who has actually shipped experiments.** SRM before anything else; post-treatment variable kept out of the balance check; a statistically significant guardrail correctly dismissed on effect size; a borderline segment correctly refused. Any one of those could be memorised. All four together is not.

5. **The recommendation is a decision, not a hedge.** It names a model, concedes the competitor is more accurate, quantifies the gap in money, states the constraints that outweigh it, and names the condition that would flip the answer.

**What I would probe in the interview** (see Part C for the questions I would ask):

- The choice to ship the *less accurate* model. I agree with it, but I want to hear them defend it under pressure, and I want to know they would change their mind if the gap widened.
- Tuning on a 60k subsample. Defensible for a time-boxed exercise, but I want to know they know what they traded away.
- They mention XGBoost handles NaN natively and then impute anyway inside the shared tree pipeline. Minor, but I want to hear them notice it.
- The two-stage expected-loss design is described but never assembled end to end. What would that pipeline actually look like in production?

**What would have made it a no-hire:** shipping `retry_within_10min_flag` in the final model, reporting accuracy as the headline, or leaving the threshold at 0.5 without comment. None of those happened.

---
---

# PART B — Design Decisions: What, Why, and What Else Was On The Table

## B.1 Cleaning and preparation

| Decision | Why | Alternatives considered | Why this one |
|---|---|---|---|
| **Drop duplicate `txn_id`** rather than keep | A duplicated row is one event recorded twice, not two observations. It double-weights that row and, if the twins straddle the split, inflates the test score | Keep them; aggregate them | At 0.6% of rows there is nothing to lose by dropping, and the alternative has a real correctness cost |
| **`customer_age` sentinels (0, 999) → `NaN`, then impute** | The row is valid; only that field is unrecorded | Drop the rows; treat 999 as a real value | Dropping loses 30 good features to save one bad one. Keeping 999 poisons the mean and every distance |
| **Negative amounts → drop the rows** | These are refunds that leaked into the payments feed — not payment attempts at all | Take the absolute value; impute | There is nothing to impute: the row does not belong in this table. Taking `abs()` would silently invent a payment that never happened |
| **Median imputation, fitted in-pipeline** | Robust to the skew that mean imputation would inherit; fitted after the split so no test information leaks | Mean; KNN-imputer; iterative/MICE; drop rows | Missingness was tested and looked uninformative, so the sophisticated imputers buy nothing for real cost. Dropping ~18% of rows would have been the actual mistake |
| **No `is_missing` indicator columns** | Failure rate was statistically indistinguishable between present and missing for all three columns | Add them anyway "just in case" | Adding three uninformative columns is three more chances to fit noise. If the rates had differed, the indicators would have gone in |
| **Cap latency at a 3,000 ms domain limit — for linear/KNN only** | ~4% of readings are instrument glitches with no relation to the outcome | Delete those rows; leave them; cap at the 99th percentile; log-transform | Deleting throws away 4% of otherwise good rows, and the IQR fence would also take out the 2% that are genuinely slow and highly predictive. **A 99th-percentile cap fails here** because the contamination is 4%, so the cap lands inside the glitch cluster. The gateway's 3-second timeout is a physical ceiling, so it is the right rule for this column. Monetary columns, whose tails are genuine, keep the percentile rule. Trees need none of this |
| **Drop `merchant_gmv_per_txn_30d`** (r = 0.95 with ticket size, VIF ≈ 11) | Two measurements of the same quantity make the coefficient split between them arbitrary | Keep both; PCA them into one component; ridge-only | PCA costs interpretability, which is the main reason logistic regression is here at all. Dropping one keeps a coefficient you can read |
| **Drop `merchant_id` (1,500 levels)** | It is an identifier; one-hot would produce 1,500 columns that memorise merchants | Target-encode it; keep as label-encoded | Its useful content is already summarised in tenure, success rate and ticket size. Target encoding on an identifier is a leakage risk that needs out-of-fold machinery to be safe |

## B.2 Encoding

| Choice | Why | Alternatives | Trade-off accepted |
|---|---|---|---|
| **One-hot** for all categoricals | Model-agnostic, no ordinality implied, no leakage risk, works identically across all five models so the comparison is like-for-like | Label/ordinal encoding; target/mean encoding; frequency encoding; XGBoost native categorical; embeddings | Label encoding invents a false ordering that a *linear* model will read as real (trees survive it). Target encoding is the strongest option for the 45-level column but needs out-of-fold computation to avoid leakage — real complexity for a modest gain. Frequency encoding is a reasonable middle ground and is what I would try next |
| **`handle_unknown="ignore"`** | A bank code appearing only in test must not crash inference | `error`; `infrequent_if_exist` | In production, unseen levels *will* happen. `infrequent_if_exist` (grouping rare levels into "other") is arguably better and is the natural follow-up |
| **Keep `issuer_bank_code` one-hot for trees and linear; ablate it for KNN** | Only distance-based models are structurally damaged by 44 sparse dimensions | Drop it everywhere; group rare levels | Dropping everywhere would discard the four or five banks that genuinely differ. The ablation cost one cell and settled the question with evidence |

## B.3 Models

| Model | Why it is in the comparison | Its actual role here |
|---|---|---|
| **Logistic regression** | The right baseline for a binary target: fast, calibrated in principle, coefficients you can read and audit, and — critically — its *failures are diagnostic* | Baseline first, then the shipped model once the features were reshaped |
| **Decision tree** | The interpretability reference and the cheapest possible demonstration of overfitting | Never a candidate to ship; earns its place as the "what does one tree look like" reference and the overfitting exhibit |
| **Random Forest** | Bagging: many decorrelated deep trees averaged. Low variance, almost no tuning needed to get something reasonable, robust to outliers and monotone transforms | Strong middle performer; the "works out of the box" benchmark |
| **XGBoost** | Boosting: each tree corrects the previous ensemble's residuals. The strongest general-purpose learner on tabular data, and the one that finds interactions without being told | Best raw accuracy; kept as the standing challenger even though it is not the one shipped |
| **KNN** | A completely different inductive bias (local, non-parametric, no training) and the model most sensitive to preprocessing — which makes it the best teaching instrument in the set | Weakest performer, included to make the scaling and dimensionality arguments *measurable* rather than asserted |
| **Linear / Ridge regression** | The stretch target `revenue_loss_inr` is continuous, and its skew is the whole lesson | Demonstrates the target-transform decision and the heteroscedasticity check |

**Not used, and why:**

- **SVM** — kernel SVMs scale roughly O(n²)–O(n³); at 160,000 rows that is impractical, and LinearSVC would add nothing that regularised logistic regression does not already give, minus the probabilities.
- **Naive Bayes** — its conditional-independence assumption is badly violated here (latency and load are correlated *and* interacting), and it is famously poorly calibrated.
- **Neural network** — no. On tabular data at this size, gradient-boosted trees match or beat MLPs while being faster, easier to tune and far easier to explain. The complexity would buy nothing.
- **SMOTE / oversampling** — 16% positives is moderate, not extreme. `class_weight` and `scale_pos_weight` achieve the same reweighting without inventing synthetic rows, and synthetic minority rows in a fraud-adjacent domain are a genuinely bad idea (they interpolate between real attack patterns to produce ones that never occur).
- **Stacking / blending** — would likely add a little AUC and would forfeit every operational reason the logistic model was chosen. Not worth it here.

## B.4 Metrics

| Metric | Used for | Why not the others |
|---|---|---|
| **PR-AUC (average precision)** | Headline model-selection metric, and the CV scorer | Focuses entirely on the positive class. At 16% positives it separates the models far more sharply than ROC-AUC does |
| **ROC-AUC** | Reported alongside for comparability | Flattered by the 84% negative class; moves slowly; the industry default, so it has to be there |
| **Precision / recall at a chosen threshold** | The operational numbers | The pair the operations team actually experiences |
| **Expected cost (₹)** | The final decision metric | The only metric denominated in the thing the business optimises |
| **Brier score + calibration curve** | Probability quality | AUC is invariant to any monotone rescaling of the scores, so it cannot detect miscalibration at all |
| **Accuracy — deliberately excluded** | — | 84% by predicting "never fails". Reporting it would be actively misleading |
| **F1 — computed but rejected as the objective** | — | Weights precision and recall equally. Here they differ by 15×, so optimising F1 optimises the wrong thing (and indeed lands on a threshold of 0.71 instead of 0.28) |

## B.5 Statistical tests

| Test | Where | Why that test |
|---|---|---|
| **Chi-square goodness of fit** | SRM check | Tests observed vs expected counts in the two arms |
| **Chi-square test of independence** | Categorical ↔ target; assignment balance | The standard test for association between two categoricals |
| **Cramér's V** | Alongside every chi-square | At n = 200,000 almost everything is "significant". V is the effect size that says whether it matters |
| **Two-proportion z-test** | Primary A/B metric | The exact test for a difference in two binomial proportions at large n. Pooled SE for the test, unpooled for the CI — pooled assumes $H_0$, which is right for the test statistic and wrong for the interval |
| **Welch's t-test** | Latency guardrail | Compares means without assuming equal variances. Costs essentially nothing when variances *are* equal, so there is no reason to prefer Student's |
| **Mann-Whitney U** | Latency guardrail, second opinion | Rank-based, distribution-free. Latency is right-skewed; running both means the conclusion does not depend on a distributional assumption |
| **Cohen's d** | Latency guardrail | The effect size that reveals a p-value of 6×10⁻⁵⁷ describes a 6 ms difference |
| **Bonferroni correction** | Segment analysis | Controls family-wise error across 6 tests. Conservative — **Benjamini–Hochberg** (false discovery rate) would be the better choice with many more segments, and is the natural next step |
| **VIF** | Multicollinearity | Quantifies how much each coefficient's variance is inflated by the others; more informative than a pairwise correlation because it catches multi-way dependence |
| **Skewness / kurtosis** | Regression target | Decides the transform question before a line of modelling code is written |
| **corr(&#124;residual&#124;, fitted)** | Regression diagnostics | A one-line, assumption-free heteroscedasticity check; Breusch–Pagan is the formal version |
| **Duan's smearing estimator** | Log-target back-transformation | Corrects the retransformation bias so the prediction targets the conditional mean rather than the median |

---
---

# PART C — Interview Questions

## C.1 Project-specific — expect these first

**Q1. Walk me through how you found the leakage.**

Three columns had names that implied they were written *after* the outcome. I quantified each one's single-feature AUC: `gateway_error_code` 0.99, `retry_within_10min_flag` 0.92, `support_ticket_raised` 0.70 — against 0.64 for the best legitimate feature. A single binary column beating every real feature by 30 points of AUC is not a great feature, it is the answer. Then I checked the mechanism: an error code exists *because* the payment failed; a retry happens *after* the customer sees the failure. At scoring time — before we route — all three are unavailable or constant. I dropped them. In a real job the next step is a five-minute conversation with the payments engineer to confirm exactly when each column is written.

**Q2. Your logistic regression started at 0.820 AUC and the Random Forest got 0.913. Most people would stop there. Why didn't you?**

Because a 0.09 gap between a linear model and a tree ensemble is a *specific* signal, not a general one. Trees differ from linear models in exactly two capabilities: they represent non-linear shapes and they capture interactions automatically. So a large gap says "there is curvature or interaction in this data that my linear model cannot express." That is a hypothesis I can test. I went back to the binned-mean plots, found a U-shape in `session_duration_sec` and an interaction between latency and load, added a squared term and a product term, and logistic regression went to 0.929. The gap was measuring my feature engineering, not the model's capacity.

**Q3. `session_duration_sec` had a correlation with the target of 0.01. Why didn't you drop it?**

Because Pearson correlation measures *linear* association only, and the relationship is U-shaped. Failure risk bottoms out around 90 seconds and rises in both directions — very fast sessions look like automated card-testing, very slow ones hit the OTP timeout. The two halves have opposite slopes and cancel to nearly zero. It turned out to be the most important feature in the dataset: shuffling it costs 0.19 of test AUC in the permutation-importance check. This is why I bin and plot every numeric feature against the target rather than trusting a correlation matrix.

**Q4. Why ship the model that is *less* accurate?**

Because the accuracy difference is small and the operational difference is not. XGBoost is ahead by about 0.005 ROC-AUC and 0.015 PR-AUC; I converted that into expected rupee cost at each model's own optimal threshold, and the gap is a few percent. Against that, the logistic model is a dot product in the routing hot path where milliseconds cost money, it has a train/test gap of 0.002 versus 0.012, and when risk asks why a transaction was declined I can hand them a coefficient table instead of a SHAP job. I would flip the decision if scoring became asynchronous, or if the cost gap widened past a few percent — and I would keep XGBoost trained and monitored regardless, because a widening gap is a free signal that new structure has appeared that my hand-built terms no longer capture.

**Q5. You capped outliers for logistic regression but not for the Random Forest. Defend that.**

A tree split asks `latency > 400?`. Whether the value on the far side is 6,000 or 25,000 changes nothing about which side of the split the row lands on — trees are invariant to any monotonic transform of a feature. Logistic regression multiplies the feature by a coefficient, so after standardisation a 25,000 ms reading becomes a ~15σ input that dominates the fitted coefficient. KNN is worse still, because that one dimension dominates the Euclidean distance. So the treatment follows the model's mechanism. Capping for the forest would not *hurt*, but doing it and calling it necessary would show I had learned the rule without the reason.

**Q5b. Why a fixed 3,000 ms cap for latency instead of the 99th percentile you used elsewhere?**

Because the contamination rate is 4%, and a 99th-percentile cap only removes the top 1% — it lands *inside* the glitch cluster and leaves most of it untouched. A percentile cap is only the right tool when you already know the contamination is smaller than the tail you are cutting. Here I had a better rule available: the gateway's own timeout is 3 seconds, so nothing above ~3,000 ms can be a real reading. That makes the cap a physical ceiling rather than an estimate, which also means it needs no train-only discipline, because it is not learned from the data at all. The monetary columns are the opposite case — their tails are genuine, so there is no domain ceiling to appeal to and the percentile rule is right there.

**Q5c. Why not just delete everything above the IQR fence?**

Because the fence flags 6% of rows and those 6% are two different populations. About 2% are genuinely slow — between the fence and 5,000 ms — and they fail at 33%, roughly double the base rate. The other 4% are above 5,000 ms and fail at 16.7%, statistically indistinguishable from the 16.0% base rate. Deleting on the fence would throw away the most predictive slice of the data along with the junk. "Outlier" is a description, not a treatment plan; you have to find out what the outliers *are* first.

**Q6. Your A/B test gave p = 3×10⁻¹⁰. Why bother with a confidence interval?**

Because the p-value only answers "is it chance?" and the CI answers "how big is it?" — which is the question the shipping decision actually depends on. The interval was [0.70, 1.34] percentage points. The whole interval sits above the 0.5 pp bar the team set, so the decision is unambiguous *and* I can hand finance a range for the annual impact rather than a single number that overstates what the experiment knows. And at n = 200,000, significance is nearly free: the latency guardrail was significant at p = 6×10⁻⁵⁷ for a 6 ms difference nobody can perceive.

**Q7. The `smart_retry` treatment increased latency and the increase was highly significant. Why did you pass the guardrail?**

Effect size. The difference is 6 ms on a 250 ms baseline — 2.4%, Cohen's d ≈ 0.07, below the conventional "small effect" threshold of 0.2. With ~190,000 observations the test can detect differences far smaller than anyone can perceive or than any SLA cares about. The larger the sample, the smaller the difference a test will flag, so at large n the effect size does the deciding and the p-value is close to a formality. A 6 ms cost for a 1 percentage-point reduction in failures is an easy trade.

**Q8. Debit Card showed a 1.09 pp lift at p = 0.014. Why didn't you report it as a win?**

Because I ran six segment tests. With six independent tests at α = 0.05, the chance of at least one false positive is 1 − 0.95⁶ ≈ 26%. The Bonferroni threshold is 0.05/6 = 0.0083 and 0.014 does not clear it. UPI does, comfortably, and it has a mechanism — UPI has the most acquirer choice, so rerouting on issuer health has the most to work with. The honest framing is that Debit Card is a **hypothesis for a follow-up experiment**, not a finding. Segments named in advance are results; segments discovered afterwards are hypotheses.

**Q9. Why did you exclude `network_latency_ms` from the covariate-balance check?**

Because it is a **post-treatment** variable — it is measured during the routing decision the treatment changes. Balance checks exist to verify that randomisation equalised things that were fixed *before* assignment. Testing a post-treatment variable for balance is a category error: an imbalance there is the treatment effect, not a randomisation failure. The same logic is why post-treatment variables must never be used as controls when estimating a treatment effect — conditioning on them blocks part of the causal path you are trying to measure, which is the collider/mediator bias problem.

**Q10. You expected dropping the 45-level categorical to help KNN. It didn't. What happened?**

It gained 0.0004 AUC, which is noise. My reasoning for checking was right but I had the magnitude wrong. Those 44 dummies are *sparse*: for any pair of transactions they contribute 0 if the banks match and 2 to the squared distance if they don't — that is close to a constant offset applied to almost every pair, not something that reorders neighbours. The curse of dimensionality bites hardest when the noise dimensions are dense and continuous, or when there are few informative dimensions to start with. Here 22 standardised numeric features are already carrying the distance. What *did* matter enormously was scaling: forgetting it cost 0.148 of AUC.

**Q10b. Your coefficient table puts five issuer-bank dummies in the top 15, but permutation importance ranks that column near the bottom. Which is right?**

Both. They answer different questions. The coefficients are correct that those five banks genuinely differ — the odds ratios are above 2 and they are the right banks. Permutation importance is correct that the column barely matters overall, because each of those banks carries about 2.2% of traffic, so a large per-unit effect on 2% of rows moves the ranking of the other 98% very little. Standardising a sparse one-hot column amplifies the appearance: a rare dummy has small variance, so dividing by it inflates the coefficient's scale. The practical rule is to read coefficients for **direction and mechanism** and permutation importance for **how much it matters** — and never to rank features by raw coefficient magnitude on one-hot columns.

**Q10c. The interaction you found reverses sign. Explain it.**

At low gateway load, slower transactions fail *less* — 22.5% down to 11.2%. At high load, slower transactions fail *far more* — 11.2% up to 27.7%. That sign flip is why `network_latency_ms` has a single-feature AUC of 0.52, indistinguishable from a coin flip: marginally the two regimes cancel exactly. It is also why an additive model cannot use the feature at all — averaging two opposite slopes gives approximately zero. My working interpretation, which I would take to engineering rather than assert, is that at low load a slow response means the issuer is running full 3-D Secure checks, which succeed, while at high load latency is queueing and queueing ends in timeouts. Same measurement, two different underlying causes — which is exactly the situation where a single coefficient is meaningless and either an interaction term or a tree is required.

**Q11. Your test set is a random split. Is that right for this problem?**

Not strictly. For a model that will score future transactions, a time-based split — train on months 1–4, test on 5–6 — is the more honest simulation, because it also catches drift. I checked the monthly failure rate and it is flat across the window, so the two should agree here, and I used the random split so the model comparison isolates model capability rather than mixing in a drift effect. In production I would use the time split as the headline number and go further with a rolling-origin backtest. It is on my before-shipping checklist for exactly that reason.

**Q12. Walk me through the feedback loop risk.**

Once we start rerouting high-risk transactions, those transactions stop failing. The next training set therefore no longer contains the failures the model prevented, so the model gradually unlearns the very patterns that made it valuable — and its measured performance degrades in a way that looks like drift but is self-inflicted. The standard mitigation is a small holdout: route 1–2% of traffic under the old policy so you keep collecting unbiased labels. It is the most under-appreciated risk in any model that changes the process generating its own training data.

## C.2 General machine learning

**Q13. Bias–variance trade-off — explain it with something from this project.**

Bias is error from a model too simple to represent the truth; variance is error from a model so flexible it fits the noise in the training sample. The unregularised decision tree is pure variance: train AUC 1.000, test AUC 0.724. The raw-feature logistic regression is bias: train 0.820, test 0.820 — almost no gap, but both are low because a straight line cannot represent a U. The two fixes are different in kind. You reduce variance with more data, regularisation, or ensembling; you reduce bias with a richer model or better features. Adding the squared term reduced *bias*, which is why it lifted train and test together.

**Q14. Bagging vs boosting.**

Bagging (Random Forest) trains many deep trees in parallel on bootstrap samples with a random feature subset per split, then averages them. Each tree is low-bias and high-variance; averaging decorrelated trees cancels the variance. It is hard to overfit by adding trees, parallelises trivially, and needs little tuning.

Boosting (XGBoost) trains trees sequentially, each one fitting the errors the current ensemble still makes. It attacks bias, so it typically reaches higher accuracy — but it *can* overfit if you add too many rounds, and it is more sensitive to hyperparameters, especially the learning rate. Here that shows up exactly as theory predicts: XGBoost wins on accuracy, the Random Forest carries the larger train/test gap out of the box, and XGBoost's gap shrank once `subsample`, `colsample_bytree` and `reg_lambda` were tuned.

**Q15. When do you scale features, and when is it pointless?**

Scale for anything that uses distances or gradient descent: KNN, SVM, k-means, PCA, neural networks, and regularised linear models — in the last case because L1/L2 penalise coefficients, and an unscaled feature gets a large coefficient purely because its units are small, so the penalty falls unevenly. Do not scale tree-based models: a split is a threshold on one feature, and scaling relabels the threshold without changing which rows fall on each side. Scaling a Random Forest is harmless but reveals a misunderstanding if you claim it was necessary.

**Q16. How do you detect and prevent target leakage?**

Detection: (1) an implausibly high score — 0.99 AUC on a hard problem means look for the leak, not celebrate; (2) single-feature AUC — any one feature far ahead of the rest is suspect; (3) read every column name and ask when the value is written; (4) check missingness patterns that mirror the target, as `gateway_error_code` did here.

Prevention: split before you do anything that learns from data; put every fitted transformation inside a `Pipeline` so cross-validation refits it per fold; use time-based splits when the problem is temporal; and for each feature ask the one question that matters — *would this value, with this value, be available at the moment I need to score?*

**Q17. Cross-validation — why, and which flavour?**

A single train/test split gives one estimate with real variance; k-fold gives k estimates and a standard error, which is what you need to know whether a 0.003 difference between two models is real. Use **StratifiedKFold** for imbalanced classification so every fold has the same class ratio. Use **TimeSeriesSplit** for temporal data so you never train on the future. Use **GroupKFold** when rows cluster — here, transactions from the same merchant, which is a real weakness of my random split. And always cross-validate on train only, refitting the preprocessing inside each fold.

**Q18. How do you handle class imbalance?**

First ask whether it is actually a problem — at 16% it is moderate, and the main consequence is that accuracy becomes useless as a metric. The lightest fixes come first: `class_weight="balanced"` or `scale_pos_weight`, which reweight the loss without touching the data. Then threshold tuning, which is free and often the largest single win. Only at extreme ratios would I reach for resampling, and I would prefer undersampling the majority (cheap, fast) to SMOTE (which invents rows that never occurred — a particularly bad idea in fraud-adjacent domains, where you would be interpolating between real attack patterns). Whatever you do, resample **inside** the CV fold, never before the split.

**Q19. Explain regularisation.**

L2 (ridge) adds λ·Σβ² to the loss, shrinking all coefficients smoothly toward zero without eliminating any — it is what keeps the 45 correlated issuer dummies from producing wild estimates. L1 (lasso) adds λ·Σ|β|, whose corner at zero drives some coefficients exactly to zero, so it performs feature selection. Elastic net mixes both, and is the right choice when you want sparsity but have correlated groups, since lasso alone arbitrarily picks one member of a correlated group. In trees the equivalent levers are `max_depth`, `min_samples_leaf`, and in XGBoost `reg_lambda`, `gamma` and the learning rate. In all cases you are trading a little bias for a lot of variance reduction.

**Q20. Precision, recall, F1 — and when is each right?**

Precision = TP/(TP+FP): of the ones I flagged, how many were real. Recall = TP/(TP+FN): of the real ones, how many did I catch. F1 is their harmonic mean, which weights them equally. Optimise recall when a miss is expensive — disease screening, or this project, where a missed failure costs ₹450 and a false alarm ₹30. Optimise precision when acting is expensive or annoying — sending a fraud team to a customer's door. Use F1 only when the two costs really are comparable; here they differ by 15×, which is why F1 lands on a threshold of 0.71 while the cost function lands on 0.28.

**Q21. Your model has 0.93 AUC offline and 0.78 in production. What went wrong?**

In rough order of likelihood: (1) **training-serving skew** — a feature computed differently in the warehouse than in the live service, which is why shadow-running is on my checklist; (2) **leakage** I missed offline, so a feature that was informative in training is constant at inference; (3) **drift** — the population moved, checkable with PSI on each input; (4) **train/test contamination**, e.g. duplicates straddling the split; (5) **the feedback loop** from Q12, if the model has been acting for a while. I would start by comparing the *input distributions* between offline and online, not the predictions — the cause is almost always upstream of the model.

## C.3 Statistics and experimentation

**Q22. What is a p-value, precisely?**

The probability of observing a test statistic at least as extreme as the one you got, *assuming the null hypothesis is true*. It is not the probability that the null is true, not the probability the result was chance, and not a measure of effect size. A p-value of 6×10⁻⁵⁷ described a 6 ms latency difference in this project, which is the cleanest possible illustration of the last point.

**Q23. Type I and Type II errors, and how do you control them?**

Type I is a false positive — rejecting a true null; you control it with α, conventionally 0.05, and with corrections like Bonferroni or Benjamini–Hochberg when you run many tests. Type II is a false negative — failing to reject a false null; you control it with power (1 − β, conventionally 0.80), which you buy with sample size, a larger effect, or lower variance. They trade off: lowering α raises β at fixed n. In business terms, a Type I error ships something that does not work; a Type II error kills something that did — and nobody writes a post-mortem about the second kind, which is why the power calculation belongs *before* the experiment.

**Q24. How would you size this experiment?**

Four inputs: baseline rate (16.5%), the minimum effect worth detecting (0.5 pp, which the team sets from the value of the win), α = 0.05 and power = 0.80. Then n per arm ≈ 2·p(1−p)·(z_{α/2} + z_β)² / MDE² ≈ 87,000. We had ~100,000 per arm, giving an MDE of 0.47 pp, and observed 1.02 pp — comfortably powered. Note the inverse-square relationship: halving the MDE quadruples the required sample. That is why detecting 0.25 pp would have needed 346,000 rows per arm.

**Q25. What is peeking, and why is it a problem?**

Checking results repeatedly and stopping as soon as p < 0.05. Each look is another chance to cross the threshold, so the actual false-positive rate rises far above the nominal 5% — with daily checks over a couple of weeks it can exceed 20%. A fixed-horizon test is only valid at a fixed horizon. If you need to monitor continuously, use methods built for it: group-sequential boundaries (O'Brien–Fleming, Pocock), always-valid p-values / confidence sequences, or a Bayesian formulation where continuous monitoring is not a violation.

**Q26. Sample Ratio Mismatch — what is it and why check it first?**

You expect a 50/50 split; you observe 50.5/49.5. A chi-square goodness-of-fit test says whether that gap is plausible by chance. If it is not, the randomiser is broken — a bot filter dropping treatment traffic, a redirect failing on one arm, a logging bug — and the units in the two arms are no longer exchangeable, which invalidates every downstream number. It costs one line of code and it catches a large share of real experiment failures, so it comes before you look at the metric.

**Q27. Correlation is not causation — how does that show up here?**

Section 4 found that transactions with sessions far from 90 seconds fail more. That is correlation. It does not license "force checkouts to 90 seconds", because the causal arrow could run either way or come from a confounder — bot traffic produces both short sessions and failures, so intervening on session length would do nothing. The only causal claim in the whole notebook is the A/B result, and it is causal *only* because assignment was randomised, which breaks the link between treatment and every confounder, observed or not. That is what randomisation buys and why an experiment beats any amount of observational modelling for a "does X cause Y" question.

**Q28. When would you not use a t-test?**

When the data are badly skewed *and* the sample is small, so the CLT has not kicked in — use Mann-Whitney or a bootstrap. When observations are not independent (repeated measures, clustered data) — use a paired test, a mixed model, or clustered standard errors. When the outcome is binary — use a proportions test or chi-square. When you have more than two groups — use ANOVA, or you reintroduce the multiple-comparisons problem. And prefer Welch's over Student's by default: it does not assume equal variances and costs essentially nothing when they happen to be equal.

**Q29. Why did you log-transform the regression target, and what did it cost you?**

Because the target had skew ≈ 6.7 and kurtosis ≈ 117, which breaks OLS's normality and homoscedasticity assumptions. I measured that rather than assuming it: the correlation between the absolute residual and the fitted value was 0.42 on the raw scale and 0.09 on the log scale. The log model halved the median error. What it cost was RMSE — `exp()` amplifies, so a moderate error in log space becomes an enormous error in rupees for the few transactions placed high in the tail. Whether the transform "wins" depends on whether you are paid to minimise typical error or worst-case error. It also introduces retransformation bias: `exp(ŷ) − 1` returns the conditional median, not the mean, which is what Duan's smearing estimator corrects — and, tellingly, applying that correction made my MAE *worse*, because MAE rewards predicting the median.

**Q30. If you had another week on this, what would you do?**

Five things, in priority order:

1. **Re-run everything on a time-based split** and add `GroupKFold` on `merchant_id`, since transactions from one merchant are not independent and my current CV slightly overstates confidence.
2. **Target-encode `issuer_bank_code` out-of-fold** and see whether it beats one-hot; it is the encoding most likely to move the number.
3. **Assemble the two-stage expected-loss pipeline end to end** — P(fail) × E[loss | fail] — and re-derive the routing decision from expected rupees rather than a probability threshold. That is the version the business actually wants.
4. **Add SHAP for the XGBoost model**, which would remove the main operational objection to shipping it and might change my recommendation.
5. **Design the follow-up experiment for the Debit Card segment** as a pre-registered hypothesis, so the finding can either be confirmed properly or dropped.
