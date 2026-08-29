"""
Tests for src/features.py, run against the real CC_GENERAL.csv dataset.

The missing-value counts below are the real, documented numbers for this
dataset (CREDIT_LIMIT=1, MINIMUM_PAYMENTS=313) -- asserted exactly, not
approximately, because they don't change between runs.
"""

import os

import pytest

from src.features import build_features, impute_missing, load_raw_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "CC_GENERAL.csv")


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data(DATA_PATH)


def test_raw_shape(raw_df):
    assert raw_df.shape == (8950, 18)


def test_real_missing_value_counts(raw_df):
    missing = raw_df.isnull().sum()
    assert missing["CREDIT_LIMIT"] == 1
    assert missing["MINIMUM_PAYMENTS"] == 313
    other_cols = [c for c in raw_df.columns if c not in ("CREDIT_LIMIT", "MINIMUM_PAYMENTS")]
    assert missing[other_cols].sum() == 0


def test_impute_missing_leaves_no_nulls(raw_df):
    imputed = impute_missing(raw_df)
    assert imputed.isnull().sum().sum() == 0


def test_impute_missing_uses_median(raw_df):
    expected_credit_limit_median = raw_df["CREDIT_LIMIT"].median()
    expected_min_payments_median = raw_df["MINIMUM_PAYMENTS"].median()

    imputed = impute_missing(raw_df)
    was_missing_credit_limit = raw_df["CREDIT_LIMIT"].isnull()
    was_missing_min_payments = raw_df["MINIMUM_PAYMENTS"].isnull()

    assert (imputed.loc[was_missing_credit_limit, "CREDIT_LIMIT"] == expected_credit_limit_median).all()
    assert (
        imputed.loc[was_missing_min_payments, "MINIMUM_PAYMENTS"] == expected_min_payments_median
    ).all()


def test_impute_missing_does_not_mutate_input(raw_df):
    before = raw_df.isnull().sum().sum()
    impute_missing(raw_df)
    after = raw_df.isnull().sum().sum()
    assert before == after


def test_build_features_shape_and_columns():
    df, features = build_features(DATA_PATH)
    assert features.shape == (8950, 13)
    assert features.isnull().sum().sum() == 0
    assert list(features.columns) == [
        "BALANCE",
        "PURCHASES",
        "ONEOFF_PURCHASES",
        "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE",
        "PURCHASES_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
        "PURCHASES_TRX",
        "CREDIT_LIMIT",
        "PAYMENTS",
        "MINIMUM_PAYMENTS",
        "PRC_FULL_PAYMENT",
        "TENURE",
    ]
