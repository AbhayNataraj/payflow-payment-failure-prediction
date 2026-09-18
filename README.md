# Payment failure prediction and a smart-retry A/B test

An end-to-end analytics project on 200,000 payment-gateway transactions. It answers two
questions that need two different methods:

1. **Did the change we shipped work?** A randomised A/B test of a new retry-routing policy,
   with hypothesis tests, a power calculation and a correction for multiple comparisons.
2. **Which payment will fail next?** A classification model comparing logistic regression,
   a decision tree, random forest, XGBoost and KNN.

The dataset is generated rather than real, and it has deliberate problems built into it:
target leakage, two near-identical columns, a 45-level categorical that's mostly noise,
sensor glitches disguised as outliers, a U-shaped relationship, an interaction that reverses
sign, and 16% class imbalance. `generate_dataset.py` documents all of it.

The notebook is committed with its outputs, so you can read the whole analysis and all seven
charts straight from GitHub without running anything.

---

## Results

### The A/B test

|  | Control | Smart retry |
|---|---:|---:|
| Payments | 99,772 | 99,578 |
| Failure rate | 16.53% | 15.49% |

Failures dropped by **1.03 percentage points**, 95% CI [0.71, 1.35], z = 6.28, p = 3.3e-10.
That's 6.2% fewer failures, worth roughly **₹1.88M a year** at full rollout (range ₹1.29M to
₹2.47M).

Two things I'd want said about that result:

- **The guardrail passed, but only on effect size.** Smart routing added 6.1ms of latency and
  the t-test came back at p = 6e-57. Cohen's d is 0.073, which is below the threshold for a
  "small" effect. With 190,000 rows a test will flag differences nobody could notice, so the
  effect size does the deciding.
- **The win is concentrated in UPI** (1.62pp, survives a Bonferroni correction). Debit Card
  looks significant on its own at p = 0.014, but across 6 segment tests the corrected
  threshold is 0.0083, so it doesn't clear. It's written up as something to test next, not
  as a finding.

### The models

Held-out test set, 39,870 rows.

| Model | ROC-AUC | PR-AUC | Train minus test AUC |
|---|---:|---:|---:|
| XGBoost | 0.9324 | 0.7857 | 0.032 |
| Logistic regression (with engineered features) | 0.9285 | 0.7737 | 0.002 |
| Random forest | 0.9127 | 0.7167 | 0.043 |
| Decision tree | 0.8810 | 0.6598 | 0.009 |
| KNN (k=35, scaled) | 0.8413 | 0.5400 | n/a |
| Logistic regression (raw features) | 0.8196 | 0.5329 | 0.001 |

**The interesting part is the first and last rows.** Raw logistic regression sits 0.11 AUC
behind XGBoost. Adding three columns, a squared term, an interaction and a log, closes 97% of
that gap. None of the three contain new information, they're all built from columns the model
already had.

So the gap was never about what a linear model can represent. It was about handing it the
features in the wrong shape. The trees' advantage was finding the U-shape and the interaction
on their own.

**What I'd ship:** logistic regression with the engineered features, at a cutoff of 0.27
rather than 0.5. XGBoost is more accurate, by about 0.004 ROC-AUC, and I'm giving that up for
inference speed in the routing hot path, coefficients I can hand to a risk reviewer, and a
train-minus-test gap of 0.002 against 0.032.

Moving the cutoff off the 0.5 default is worth **20% of the expected cost** and costs nothing,
because a missed failure costs about 15x a false alarm (₹450 against ₹30).

---

## Files

| File | What it is |
|---|---|
| `payment_failure_prediction.ipynb` | The analysis, run end to end with outputs |
| `payments_ab_failure_200k.csv.gz` | The dataset, gzipped (201,200 rows x 35 columns, 11 MB packed) |
| `generate_dataset.py` | The generator, with every built-in problem documented |
| `requirements.txt` | Package versions |

## Running it

```bash
git clone https://github.com/AbhayNataraj/payflow-payment-failure-prediction.git
cd payflow-payment-failure-prediction

pip install -r requirements.txt

# unpack the dataset, the notebook looks for the .csv
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('payments_ab_failure_200k.csv.gz'), open('payments_ab_failure_200k.csv','wb'))"

jupyter notebook payment_failure_prediction.ipynb
```

Takes about 7 minutes end to end on 2 cores. You can also rebuild the dataset from scratch
instead of unpacking it, the seed is fixed so it comes out byte for byte identical:

```bash
python generate_dataset.py
```

In Colab, upload the `.csv` next to the notebook and run all cells. `xgboost` is already there.

---

## What's in the notebook

**1. Cleaning.** Duplicate webhook rows, a constant column, a card network spelled five
different ways, sentinel ages of 0 and 999, refunds that leaked into the payments table. The
point isn't that they're fixed, it's that the fix differs each time depending on what's
actually wrong: age 999 is a good row with one bad field, a negative amount is a row that
doesn't belong in the table at all.

**2. Exploring.** Class balance and why accuracy is out. Then the latency "outliers", which
turn out to be two different populations: 2% genuinely slow that fail at 33%, and 4% above
5,000ms that fail at the base rate and are just broken sensor readings. Cutting at the IQR
fence would have thrown away the most predictive slice of the data.

Also the plot the project turns on: `session_duration_sec` correlates 0.21 with the target,
which reads as forgettable, but binned it's a clear U with risk climbing in both directions
from 90 seconds. And a latency-by-load heatmap where the effect of latency **reverses sign**
between low and high load, which is why latency alone has a single-feature AUC of 0.52.

**3. Leakage check.** Three columns that get written *because* of the outcome, scoring 0.99,
0.92 and 0.70 AUC on their own against 0.64 for the best legitimate feature. Keeping one of
them gives you a 0.97 AUC model that does nothing in production, because at scoring time the
column is always empty.

**4. A/B test.** Split-ratio check first, then a two-proportion z-test with a confidence
interval, a power calculation showing why the test needed 200,000 rows, business impact as a
range rather than a number, the latency guardrail, and segments with a Bonferroni correction.

**5-6. Models.** Two preprocessing recipes rather than one, because scaling and capping matter
for logistic regression and KNN and do nothing for trees. Five models, an unregularised tree
left in to show overfitting (train AUC 1.000, test 0.722), and a KNN run with and without
scaling to put a number on what forgetting it costs (0.15 AUC, silently). Then the feature
engineering.

**7. Evaluation.** ROC and precision-recall curves, a cutoff picked from a cost function
rather than left at 0.5, and the coefficients.

**8. Recommendation.** What to ship, what to check before trusting it in production, and what
the project doesn't prove.

---

## Techniques used

**Preparation:** deduplication, constant-column removal, categorical normalisation, sentinel
handling, missingness tested for informativeness before choosing a strategy, pipeline-based
imputation, train-only capping

**Statistics:** two-proportion z-test, chi-square, Welch's t-test, Cohen's d, confidence
intervals, power and minimum detectable effect, Bonferroni correction

**Machine learning:** logistic regression, decision trees, random forest, XGBoost, KNN,
`Pipeline` / `ColumnTransformer`, stratified splitting, class weighting, feature engineering
for non-linearity and interactions, cost-based threshold selection, target-leakage detection
