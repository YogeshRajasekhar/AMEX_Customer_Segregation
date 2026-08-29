# AMEX Customer Segregation

Segments 8,950 real credit card customers by behavior, then uses a contextual
bandit (LinUCB) to decide which offer to show each segment.

This is a real, already-run analysis, refactored into a proper package with
tests. The numbers below are not illustrative — they are what this code
produces when you run it. Two things that went wrong along the way are kept
in, on purpose, because they're part of the real story:

1. **DBSCAN's silhouette-optimal configuration collapsed 98% of customers
   into one cluster.** Mathematically the best-scoring result, practically
   useless.
2. **The bandit's first configuration converged to the wrong action for one
   segment.** Verified against a computed ground truth, then fixed by
   increasing exploration and re-verified correct.

## Dataset

`data/CC_GENERAL.csv` — 8,950 real credit card customers, 18 behavioral
variables aggregated over 6 months. This is the standard public benchmark
dataset for this type of project (used identically across many independent
public analyses). It has no timestamp column and no real offer-response
history — both of those limitations matter for what's below and are not
glossed over.

## Pipeline

### 1. Feature engineering (`src/features.py`)

Raw shape: `(8950, 18)`. Two columns have real missing values —
`CREDIT_LIMIT` (1) and `MINIMUM_PAYMENTS` (313) — median-imputed, the
documented standard approach for this dataset. 13 RFM-style features are
kept: monetary (`BALANCE`, `PURCHASES`, `PAYMENTS`, ...), frequency
(`PURCHASES_TRX`), and an honest recency *proxy* — this dataset has no
timestamp, so `PURCHASES_FREQUENCY` / `CASH_ADVANCE_FREQUENCY` stand in for
"how consistently active," not literal days-since. Engineered feature set:
`(8950, 13)`.

### 2. Clustering comparison (`src/clustering.py`)

Features are scaled (mandatory before any distance-based method — otherwise
`BALANCE` ranging 0–19,043 would dominate `PURCHASES_FREQUENCY` ranging 0–1
purely on scale, not real importance), and PCA is applied only because the
real explained-variance curve calls for it: **7 of 13 components are needed
for ≥85% variance.**

K-Means is run for k=2..8, scored by silhouette. **Silhouette-optimal k is 2
(score 0.4799).**

DBSCAN is grid-searched over `eps ∈ {0.5, 0.8, 1.0, 1.5, 2.0, 2.5}` and
`min_samples ∈ {5, 10, 15}`, also scored by silhouette (on non-noise points).
**The best-scoring config is `eps=2.0, min_samples=5`, silhouette 0.5184 —
higher than K-Means.** But that config puts **8,767 of 8,950 customers
(98.0%) into one cluster**, plus a handful of 4–5-person micro-clusters. It
wins on the metric and is useless for building personas. This is documented,
not dropped — see `tests/test_clustering.py::test_dbscan_silhouette_optimal_config_is_a_mega_cluster`.

### 3. Business-usable clustering

Given that the two "statistically optimal" results above (K-Means k=2,
DBSCAN's mega-cluster) are both unusable for real personas, cluster count is
picked for **business interpretability instead: k=4** — a deliberate,
documented trade of some silhouette score (0.2386 at k=4 vs. 0.4799 at k=2)
for a result a marketing team can actually act on. This is exactly
reproducible with `random_state=42`:

| Segment | Size | % |
|---|---|---|
| 0 | 3,235 | 36.1% |
| 1 | 4,319 | 48.3% |
| 2 | 1,157 | 12.9% |
| 3 | 239 | 2.7% |

### 4. Personas (`src/personas.py`)

Persona names and descriptions are derived from which real per-segment
statistic each cluster is highest or lowest on — not asserted first and
back-filled with numbers to match:

- **Segment 0 — reliable everyday transactors** (36.1%): highest
  full-payment rate, lowest balance, lowest cash-advance use.
- **Segment 1 — low-engagement, low-limit customers** (48.3%): lowest credit
  limit, lowest purchase frequency, lowest purchase volume.
- **Segment 2 — cash-advance-heavy revolvers** (12.9%): heaviest
  cash-advance use, highest balance, lowest full-payment rate.
- **Segment 3 — high-value elites** (2.7%): highest credit limit, highest
  purchase frequency, highest purchase volume.

### 5. Contextual bandit (`src/bandit.py`)

A real LinUCB implementation (Li et al., 2010) — one ridge-regression model
per offer (arm), context = segment one-hot vector, upper-confidence-bound
action selection — decides which of 4 offers
(`cashback_card`, `travel_rewards`, `low_apr_balance_transfer`,
`premium_rewards`) to show each segment.

**Honest framing: the reward signal is simulated, not real customer response
data.** This dataset is a static 6-month snapshot with no logged
interaction history, so `simulate_reward()` uses a reward function grounded
in each segment's real computed statistics (e.g. high cash-advance-frequency
segments respond better to a low-APR balance-transfer offer). This is
standard, legitimate practice for offline/simulated bandit evaluation
without live traffic — not a shortcut, and not presented as real customer
data.

#### The wrong-then-fixed convergence story

The first configuration (`n_rounds=5000, alpha=1.0`) converged to the
**wrong** action for Segment 2: it learned `cashback_card`, but the true
best action — verified directly against the reward function's structure,
in `TRUE_BEST` — is `low_apr_balance_transfer`. Root cause: Segment 2 is
only ~13% of customers, and `alpha=1.0` didn't force enough exploration to
distinguish a real but modest ~17% reward gap before the bandit committed.

Fixed by increasing exploration: `n_rounds=20000, alpha=2.5`. Re-verified
correct for all 4 segments. Both configurations are preserved and tested —
see `tests/test_bandit.py` — rather than only keeping the fixed one.

With the corrected config:

```
Random baseline conversion rate: 0.1575
LinUCB bandit conversion rate:   0.2390
Lift over random: 51.7%

Segment 0: learned=cashback_card               [CORRECT]
Segment 1: learned=cashback_card               [CORRECT]
Segment 2: learned=low_apr_balance_transfer     [CORRECT]
Segment 3: learned=cashback_card               [CORRECT]
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

## Tests

```bash
pytest tests/ -v
```

Actual output:

```
============================= test session starts ==============================
collected 15 items

tests/test_bandit.py::test_true_best_covers_all_four_segments PASSED     [  6%]
tests/test_bandit.py::test_segment_2_convergence[original-config-historical-regression] PASSED [ 13%]
tests/test_bandit.py::test_segment_2_convergence[corrected-config] PASSED [ 20%]
tests/test_bandit.py::test_corrected_config_converges_for_all_segments PASSED [ 26%]
tests/test_bandit.py::test_corrected_config_beats_random_baseline PASSED [ 33%]
tests/test_clustering.py::test_pca_reduces_to_seven_components PASSED    [ 40%]
tests/test_clustering.py::test_kmeans_silhouette_optimal_k_is_two PASSED [ 46%]
tests/test_clustering.py::test_business_kmeans_k4_exact_segment_sizes PASSED [ 53%]
tests/test_clustering.py::test_dbscan_silhouette_optimal_config_is_a_mega_cluster PASSED [ 60%]
tests/test_features.py::test_raw_shape PASSED                            [ 66%]
tests/test_features.py::test_real_missing_value_counts PASSED            [ 73%]
tests/test_features.py::test_impute_missing_leaves_no_nulls PASSED       [ 80%]
tests/test_features.py::test_impute_missing_uses_median PASSED           [ 86%]
tests/test_features.py::test_impute_missing_does_not_mutate_input PASSED [ 93%]
tests/test_features.py::test_build_features_shape_and_columns PASSED     [100%]

============================= 15 passed in 49.31s ==============================
```

`test_bandit.py::test_segment_2_convergence[original-config-historical-regression]`
is the documentation test: it asserts the original config really does
produce the wrong action for Segment 2, so that bug stays visible instead of
disappearing once it's fixed.

## Project structure

```
├── README.md
├── requirements.txt
├── data/CC_GENERAL.csv
├── src/
│   ├── features.py      # load, impute, engineer RFM-style features
│   ├── clustering.py     # scale, PCA, K-Means vs. DBSCAN, business k=4
│   ├── personas.py       # human-readable persona descriptions
│   └── bandit.py         # LinUCB contextual bandit + reward simulation
├── tests/
│   ├── test_features.py
│   ├── test_clustering.py
│   └── test_bandit.py
└── main.py                # runs the full pipeline end to end
```
