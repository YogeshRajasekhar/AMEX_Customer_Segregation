"""
Tests for src/bandit.py, run against the real k=4 clustered customer data.

The important test here is parametrized over two configurations:
- the ORIGINAL config (n_rounds=5000, alpha=1.0), which really did converge
  to the WRONG action for Segment 2 -- this is documented as a
  historical-regression test, not smoothed over;
- the CORRECTED config (n_rounds=20000, alpha=2.5), which is verified
  correct for all 4 segments and must keep passing.

Both configurations are asserted against the same TRUE_BEST ground truth
computed directly from the reward function's structure.
"""

import os

import pytest

from src.bandit import (
    CORRECTED_ALPHA,
    CORRECTED_N_ROUNDS,
    ORIGINAL_ALPHA,
    ORIGINAL_N_ROUNDS,
    TRUE_BEST,
    compute_segment_stats,
    lift_over_random,
    run_bandit,
)
from src.clustering import business_kmeans, reduce_dimensions, scale_features
from src.features import build_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "CC_GENERAL.csv")


@pytest.fixture(scope="module")
def clustered_customers():
    """Real k=4 business clustering, built the same way main.py does."""
    _, features = build_features(DATA_PATH)
    X_scaled, _ = scale_features(features)
    X_pca, _, _ = reduce_dimensions(X_scaled)
    labels, _ = business_kmeans(X_pca)
    features = features.copy()
    features["cluster"] = labels
    return features


@pytest.fixture(scope="module")
def segment_stats(clustered_customers):
    return compute_segment_stats(clustered_customers)


def test_true_best_covers_all_four_segments():
    assert set(TRUE_BEST.keys()) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "n_rounds, alpha, expect_segment_2_correct",
    [
        pytest.param(
            ORIGINAL_N_ROUNDS,
            ORIGINAL_ALPHA,
            False,
            id="original-config-historical-regression",
        ),
        pytest.param(
            CORRECTED_N_ROUNDS,
            CORRECTED_ALPHA,
            True,
            id="corrected-config",
        ),
    ],
)
def test_segment_2_convergence(
    clustered_customers, segment_stats, n_rounds, alpha, expect_segment_2_correct
):
    """
    Segment 2 is the historically interesting case: it's only ~13% of
    customers, and under the original config (n_rounds=5000, alpha=1.0)
    the bandit converged to cashback_card when the real best action is
    low_apr_balance_transfer -- verified WRONG here on purpose, documenting
    the real bug. Under the corrected config (n_rounds=20000, alpha=2.5)
    it converges CORRECTLY. Both are asserted, not just the fixed one.
    """
    customer_segments = clustered_customers["cluster"].values
    result = run_bandit(customer_segments, segment_stats, n_rounds=n_rounds, alpha=alpha, seed=42)

    learned_segment_2 = result["learned_actions"][2]
    is_correct = learned_segment_2 == TRUE_BEST[2]

    assert is_correct == expect_segment_2_correct, (
        f"n_rounds={n_rounds}, alpha={alpha}: learned={learned_segment_2}, "
        f"true={TRUE_BEST[2]}, expected correct={expect_segment_2_correct}"
    )


def test_corrected_config_converges_for_all_segments(clustered_customers, segment_stats):
    customer_segments = clustered_customers["cluster"].values
    result = run_bandit(
        customer_segments,
        segment_stats,
        n_rounds=CORRECTED_N_ROUNDS,
        alpha=CORRECTED_ALPHA,
        seed=42,
    )

    for segment, true_action in TRUE_BEST.items():
        assert result["learned_actions"][segment] == true_action, f"segment {segment}"


def test_corrected_config_beats_random_baseline(clustered_customers, segment_stats):
    customer_segments = clustered_customers["cluster"].values
    result = run_bandit(
        customer_segments,
        segment_stats,
        n_rounds=CORRECTED_N_ROUNDS,
        alpha=CORRECTED_ALPHA,
        seed=42,
    )
    lift = lift_over_random(result["bandit_rewards"], result["random_rewards"])
    assert lift > 0
    assert lift == pytest.approx(51.7, abs=1.0)
