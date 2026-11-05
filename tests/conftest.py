"""Shared fixtures."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, split_orders


@pytest.fixture(scope="session")
def split():
    """X_train, X_test, y_train, y_test from the real split."""
    return split_orders()


@pytest.fixture(scope="session")
def fitted_pipeline(split):
    """A pipeline fitted on the training set only."""
    X_train, _, y_train, _ = split
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)


@pytest.fixture
def odd_order():
    """One order with a missing prep_minutes and a weather never seen in training."""
    return pd.DataFrame(
        {
            "distance_km": [4.2],
            "items": [3],
            "hour": [19],
            "weather": ["hail"],
            "day_type": ["weekend"],
            "vehicle": ["bike"],
            "prep_minutes": [float("nan")],
            "courier_months": [12],
        }
    )
