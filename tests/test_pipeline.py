"""Tests for the feature list, the split and the pipeline."""

import numpy as np
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TEST_SIZE,
    split_orders,
)


@pytest.mark.parametrize("column", ["order_id", "customer_rating", "late"])
def test_feature_list_excludes_columns_we_decided_against(column):
    assert column not in FEATURES


def test_feature_list_has_no_duplicates_and_matches_the_data(split):
    X_train, _, _, _ = split
    assert len(FEATURES) == len(set(FEATURES))
    assert set(FEATURES) == set(X_train.columns)
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)


def test_split_is_stratified(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_test) / (len(X_train) + len(X_test)) == pytest.approx(TEST_SIZE)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.005)


def test_split_is_repeatable(split):
    X_train, X_test, y_train, _ = split
    X_train2, X_test2, y_train2, _ = split_orders()
    assert X_train.index.equals(X_train2.index)
    assert X_test.index.equals(X_test2.index)
    assert y_train.equals(y_train2)


def test_train_and_test_do_not_overlap(split):
    X_train, X_test, _, _ = split
    assert set(X_train.index).isdisjoint(X_test.index)


def test_predicted_probabilities_are_valid(fitted_pipeline, split):
    _, X_test, _, _ = split
    proba = fitted_pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_handles_missing_value_and_unseen_category(fitted_pipeline, odd_order):
    proba = fitted_pipeline.predict_proba(odd_order)[:, 1]
    assert proba.shape == (1,)
    assert 0 <= proba[0] <= 1
