"""
Generates the PayFlow payment-failure dataset (~200,000 rows).

Domain: an online payment gateway. Each row is one payment attempt.

Primary target      : payment_failed        (binary, ~16% positive)
Secondary target    : revenue_loss_inr      (right-skewed, > 0 only when a payment failed)
Experiment column   : experiment_group      (control vs smart_retry, 50/50 randomised)

The generative process deliberately contains the following properties so that the
analysis notebook has something real to discover:

  1. Target leakage        - retry_within_10min_flag, support_ticket_raised, gateway_error_code
  2. Multicollinearity     - merchant_avg_ticket_size_30d vs merchant_gmv_per_txn_30d (r ~ 0.96)
  3. High cardinality      - issuer_bank_code (45 levels, mostly noise)
  4. Missing values        - customer_age (~10%), device_trust_score (~7%), merchant_risk_score (~3%)
  5. Outliers              - network_latency_ms (~4% sensor-glitch spikes, pure noise)
  6. Non-linearity         - U-shaped effect of session_duration_sec around an optimum of ~90s
  7. Interaction           - network_latency_ms x gateway_load_pct (multiplicative)
  8. Log-saturating effect - merchant_tenure_months
  9. Class imbalance       - ~16% positive
 10. Skewed reg. target    - revenue_loss_inr is log-normal
 11. Dirty data            - duplicates, inconsistent casing, whitespace, impossible values,
                             a constant column, sentinel values
 12. A/B effect            - smart_retry lowers failure rate ~1pp, and raises latency ~6ms
                             (a small guardrail regression)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260907)
N = 200_000

# ----------------------------------------------------------------------------------
# 1. Categorical backbone
# ----------------------------------------------------------------------------------

payment_method = RNG.choice(
    ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet", "EMI"],
    size=N, p=[0.42, 0.18, 0.16, 0.11, 0.09, 0.04],
)

merchant_category = RNG.choice(
    ["Ecommerce", "Travel", "Food Delivery", "Education", "Gaming",
     "Utilities", "Healthcare", "SaaS", "Financial Services", "Retail"],
    size=N, p=[0.22, 0.10, 0.15, 0.08, 0.07, 0.09, 0.06, 0.08, 0.06, 0.09],
)

device_os = RNG.choice(["Android", "iOS", "Web", "Other"], size=N, p=[0.55, 0.22, 0.21, 0.02])
customer_tier = RNG.choice(["New", "Returning", "Loyal"], size=N, p=[0.30, 0.45, 0.25])
region = RNG.choice(
    ["West", "South", "North", "East", "Central", "North East"],
    size=N, p=[0.27, 0.26, 0.21, 0.13, 0.09, 0.04],
)

# High-cardinality, almost-all-noise categorical
issuer_bank_code = RNG.choice([f"IBK{str(i).zfill(3)}" for i in range(1, 46)], size=N)

# Card network only makes sense for card rails
card_network = np.where(
    np.isin(payment_method, ["Credit Card", "Debit Card", "EMI"]),
    RNG.choice(["Visa", "Mastercard", "RuPay", "Amex"], size=N, p=[0.38, 0.32, 0.26, 0.04]),
    "Not Applicable",
)

# ----------------------------------------------------------------------------------
# 2. Merchant-level attributes
# ----------------------------------------------------------------------------------

n_merchants = 1500
merchant_idx = RNG.integers(0, n_merchants, size=N)
merchant_id = np.array([f"MID-{str(i).zfill(4)}" for i in range(n_merchants)])[merchant_idx]

m_tenure = np.clip(RNG.gamma(shape=2.0, scale=14.0, size=n_merchants), 1, 120).round(0)
m_ticket = np.clip(RNG.lognormal(mean=7.0, sigma=0.75, size=n_merchants), 80, 90_000)
# Near-duplicate of ticket size: same quantity measured a second way -> r ~ 0.96
m_gmv_per_txn = m_ticket * RNG.normal(1.0, 0.20, size=n_merchants) + RNG.normal(0, 45, size=n_merchants)
m_gmv_per_txn = np.clip(m_gmv_per_txn, 50, 120_000)
m_success_rate = np.clip(RNG.normal(85.0, 5.5, size=n_merchants), 55, 99)

merchant_tenure_months = m_tenure[merchant_idx]
merchant_avg_ticket_size_30d = m_ticket[merchant_idx]
merchant_gmv_per_txn_30d = m_gmv_per_txn[merchant_idx]
merchant_success_rate_30d = m_success_rate[merchant_idx]

merchant_risk_score = np.clip(
    RNG.normal(38, 16, size=N) + (85 - merchant_success_rate_30d) * 0.9, 0, 100
)

# ----------------------------------------------------------------------------------
# 3. Transaction-level attributes
# ----------------------------------------------------------------------------------

txn_amount_inr = np.clip(
    merchant_avg_ticket_size_30d * RNG.lognormal(mean=0.0, sigma=0.55, size=N), 10, 500_000
)

_hour_w = np.array([2, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 4, 3, 2], dtype=float)
txn_hour = RNG.choice(np.arange(24), size=N, p=_hour_w / _hour_w.sum())

start = pd.Timestamp("2026-01-01")
day_offset = RNG.integers(0, 181, size=N)
txn_timestamp = (start + pd.to_timedelta(day_offset, unit="D")
                 + pd.to_timedelta(txn_hour, unit="h")
                 + pd.to_timedelta(RNG.integers(0, 3600, size=N), unit="s"))
is_weekend = (txn_timestamp.dayofweek >= 5).astype(int)

# U-shaped driver: very short sessions look like card-testing bots,
# very long sessions run into OTP timeouts. Optimum is around 90 seconds.
session_duration_sec = np.clip(
    RNG.normal(95, 58, size=N) + RNG.lognormal(mean=np.log(20), sigma=0.7, size=N) - 20,
    8, 330).round(1)

gateway_load_pct = np.clip(
    RNG.normal(55, 17, size=N) + np.where(np.isin(txn_hour, [19, 20, 21, 12, 13]), 12, 0), 5, 100
)

# "True" latency, i.e. the value that actually influences the outcome
latency_true = np.clip(
    RNG.lognormal(mean=np.log(170), sigma=0.42, size=N) + gateway_load_pct * 1.1, 25, 2200
)

prior_failed_attempts_24h = RNG.poisson(0.35, size=N)
retry_attempt_number = np.where(prior_failed_attempts_24h > 0,
                                RNG.integers(2, 5, size=N), 1)

customer_age = np.clip(RNG.normal(34, 11, size=N), 18, 82).round(0)
device_trust_score = np.clip(RNG.normal(68, 18, size=N), 0, 100)
is_international = (RNG.random(N) < 0.07).astype(int)
kyc_verified = (RNG.random(N) < np.where(customer_tier == "New", 0.62, 0.90)).astype(int)
card_bin_country_match = np.where(is_international == 1,
                                  (RNG.random(N) < 0.55).astype(int),
                                  (RNG.random(N) < 0.985).astype(int))

# ----------------------------------------------------------------------------------
# 4. A/B experiment assignment
# ----------------------------------------------------------------------------------

experiment_group = np.where(RNG.random(N) < 0.5, "control", "smart_retry")
is_treatment = (experiment_group == "smart_retry").astype(int)

# Guardrail: the smarter routing costs a few milliseconds of latency.
latency_true = latency_true + is_treatment * RNG.normal(6.0, 2.0, size=N)

# ----------------------------------------------------------------------------------
# 5. Ground-truth failure probability
# ----------------------------------------------------------------------------------

# (a) U-shaped / quadratic term around the ~90s optimum
q_session = np.clip((session_duration_sec - 95.0) / 60.0, -1.5, 2.6)
f_session = 1.5 * (q_session ** 2 - 1.0)

# (b) multiplicative interaction: latency only bites when the gateway is busy
lat_n = np.clip((latency_true - 250.0) / 110.0, -1.8, 3.0)
load_n = np.clip((gateway_load_pct - 55.0) / 26.0, -1.9, 1.9)
f_interaction = 2.0 * lat_n * load_n

# (c) log-saturating protective effect of merchant tenure
f_tenure = -1.1 * np.log1p(merchant_tenure_months)

# (d) roughly linear "easy" signal
f_linear = 3.2 * (
    0.55 * np.log1p(prior_failed_attempts_24h)
    + 0.16 * retry_attempt_number
    - 0.62 * kyc_verified
    + 0.55 * is_international
    - 0.55 * card_bin_country_match
    - 0.030 * (merchant_success_rate_30d - 85.0)
    - 0.0045 * (device_trust_score - 68.0)
    + 0.0035 * (merchant_risk_score - 38.0)
    + 0.10 * np.log1p(txn_amount_inr / 1000.0)
)

# (e) payment-method baseline hazards
method_effect = 2.5 * pd.Series(payment_method).map({
    "UPI": -0.18, "Credit Card": 0.10, "Debit Card": 0.22,
    "Net Banking": 0.34, "Wallet": -0.10, "EMI": 0.30,
}).to_numpy()

# (f) a handful of issuer banks are genuinely bad; the other 40 are noise
bad_issuers = {"IBK007": 0.40, "IBK019": 0.34, "IBK031": 0.45, "IBK042": 0.30, "IBK013": -0.28}
issuer_effect = 2.5 * pd.Series(issuer_bank_code).map(bad_issuers).fillna(0.0).to_numpy()

# (g) the treatment effect we want the A/B section to recover
# The lift is not uniform: smart retry helps UPI a lot, does nothing for net banking.
treat_by_method = pd.Series(payment_method).map({
    "UPI": -0.20, "Net Banking": 0.14, "Credit Card": -0.02,
    "Debit Card": -0.02, "Wallet": 0.0, "EMI": 0.0,
}).to_numpy()
f_treatment = is_treatment * (-0.14 + treat_by_method)

score = (f_session + f_interaction + f_tenure + f_linear
         + method_effect + issuer_effect + f_treatment)

# Solve for the intercept that puts the overall failure rate at ~16%
TARGET_RATE = 0.16
lo, hi = -12.0, 6.0
for _ in range(80):
    mid = (lo + hi) / 2
    if (1.0 / (1.0 + np.exp(-(score + mid)))).mean() > TARGET_RATE:
        hi = mid
    else:
        lo = mid
INTERCEPT = (lo + hi) / 2
print("solved intercept:", round(INTERCEPT, 4))

logit = score + INTERCEPT
p_fail = 1.0 / (1.0 + np.exp(-logit))
payment_failed = (RNG.random(N) < p_fail).astype(int)

# ----------------------------------------------------------------------------------
# 6. Post-outcome columns (target leakage)
# ----------------------------------------------------------------------------------

# A retry within 10 minutes almost only happens after a failure.
retry_within_10min_flag = np.where(
    payment_failed == 1, (RNG.random(N) < 0.88).astype(int), (RNG.random(N) < 0.035).astype(int)
)

support_ticket_raised = np.where(
    payment_failed == 1, (RNG.random(N) < 0.41).astype(int), (RNG.random(N) < 0.012).astype(int)
)

error_codes = ["ERR_INSUFFICIENT_FUNDS", "ERR_OTP_TIMEOUT", "ERR_ISSUER_DOWN",
               "ERR_RISK_DECLINE", "ERR_GATEWAY_TIMEOUT", "ERR_INVALID_CARD"]
gateway_error_code = np.where(
    payment_failed == 1,
    RNG.choice(error_codes, size=N, p=[0.30, 0.22, 0.16, 0.14, 0.12, 0.06]),
    np.where(RNG.random(N) < 0.02, "ERR_SOFT_DECLINE_RECOVERED", None),
)

# ----------------------------------------------------------------------------------
# 7. Regression target: revenue lost on a failed attempt (right-skewed)
# ----------------------------------------------------------------------------------

loss_mu = (np.log(np.clip(txn_amount_inr, 10, None)) * 0.62
           + 0.30 * is_international
           + 0.10 * np.log1p(retry_attempt_number)
           + 1.05)
revenue_loss_inr = np.where(
    payment_failed == 1,
    np.round(np.exp(loss_mu + RNG.normal(0, 0.85, size=N)), 2),
    0.0,
)
revenue_loss_inr = np.clip(revenue_loss_inr, 0, 900_000)

# ----------------------------------------------------------------------------------
# 8. Observed (dirty) versions of the features
# ----------------------------------------------------------------------------------

# Sensor glitches: 4% of latency readings are junk spikes that carry no signal
network_latency_ms = latency_true.copy()
glitch = RNG.random(N) < 0.04
network_latency_ms[glitch] = RNG.uniform(6_000, 25_000, size=glitch.sum())

df = pd.DataFrame({
    "txn_id": [f"TXN-{str(i).zfill(8)}" for i in range(1, N + 1)],
    "txn_timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    "txn_hour": txn_hour,
    "is_weekend": is_weekend,
    "merchant_id": merchant_id,
    "merchant_category": merchant_category,
    "payment_method": payment_method,
    "card_network": card_network,
    "issuer_bank_code": issuer_bank_code,
    "device_os": device_os,
    "region": region,
    "customer_tier": customer_tier,
    "experiment_group": experiment_group,
    "txn_amount_inr": txn_amount_inr.round(2),
    "customer_age": customer_age,
    "is_international": is_international,
    "kyc_verified": kyc_verified,
    "card_bin_country_match": card_bin_country_match,
    "device_trust_score": device_trust_score.round(1),
    "merchant_tenure_months": merchant_tenure_months.astype(int),
    "merchant_avg_ticket_size_30d": merchant_avg_ticket_size_30d.round(2),
    "merchant_gmv_per_txn_30d": merchant_gmv_per_txn_30d.round(2),
    "merchant_success_rate_30d": merchant_success_rate_30d.round(2),
    "merchant_risk_score": merchant_risk_score.round(1),
    "session_duration_sec": session_duration_sec,
    "network_latency_ms": network_latency_ms.round(1),
    "gateway_load_pct": gateway_load_pct.round(1),
    "prior_failed_attempts_24h": prior_failed_attempts_24h,
    "retry_attempt_number": retry_attempt_number,
    "data_version": "v2.1",
    # ---- post-outcome / leaky ----
    "gateway_error_code": gateway_error_code,
    "retry_within_10min_flag": retry_within_10min_flag,
    "support_ticket_raised": support_ticket_raised,
    # ---- targets ----
    "payment_failed": payment_failed,
    "revenue_loss_inr": revenue_loss_inr,
})

# ----------------------------------------------------------------------------------
# 9. Inject realistic mess
# ----------------------------------------------------------------------------------

# 9a. Missing values (missing completely at random)
for col, rate in [("customer_age", 0.10), ("device_trust_score", 0.07), ("merchant_risk_score", 0.03)]:
    df.loc[RNG.random(N) < rate, col] = np.nan

# 9b. Impossible / sentinel values in customer_age
sent = RNG.choice(N, size=1_100, replace=False)
df.loc[sent[:550], "customer_age"] = 0
df.loc[sent[550:], "customer_age"] = 999

# 9c. Wrongly-signed amounts (refund rows that leaked into the feed)
neg = RNG.choice(N, size=650, replace=False)
df.loc[neg, "txn_amount_inr"] = -df.loc[neg, "txn_amount_inr"].abs()

# 9d. Inconsistent categorical casing / padding
mask = RNG.random(N) < 0.30
df.loc[mask, "card_network"] = df.loc[mask, "card_network"].str.upper()
mask = RNG.random(N) < 0.12
df.loc[mask, "card_network"] = " " + df.loc[mask, "card_network"] + " "

mask = RNG.random(N) < 0.02
df.loc[mask, "merchant_id"] = df.loc[mask, "merchant_id"] + " "

# 9e. "unknown" masquerading as a category level
mask = RNG.random(N) < 0.015
df.loc[mask, "merchant_category"] = "unknown"

# 9f. Exact duplicate rows (double-fired webhooks)
dupes = df.sample(n=1_200, random_state=7)
df = pd.concat([df, dupes], ignore_index=True)
df = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

df.to_csv("payments_ab_failure_200k.csv", index=False)
print("rows, cols:", df.shape)
print("failure rate:", round(df["payment_failed"].mean(), 4))
print("control  fail rate:", round(df.loc[df.experiment_group == "control", "payment_failed"].mean(), 5))
print("treated  fail rate:", round(df.loc[df.experiment_group == "smart_retry", "payment_failed"].mean(), 5))
