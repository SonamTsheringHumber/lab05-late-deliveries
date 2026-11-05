"""Feature lists, the train/test split and the one pipeline used everywhere."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

# One seed for every random choice in the project.
SEED = 42
TEST_SIZE = 0.25
TARGET = "late"

# Columns known at the moment the order is placed.
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]
CATEGORICAL_FEATURES = ["weather", "day_type", "vehicle"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns deliberately left out, with the reason.
EXCLUDED_COLUMNS = {
    "order_id": "an identifier with no meaning; a flexible model could memorise it",
    "customer_rating": "given after delivery, so unknown at order time (leakage)",
    TARGET: "the target itself",
}


def split_orders(
    orders: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 75/25 split. Returns X_train, X_test, y_train, y_test."""
    if orders is None:
        orders = make_orders()
    return train_test_split(
        orders[FEATURES],
        orders[TARGET],
        test_size=TEST_SIZE,
        stratify=orders[TARGET],
        random_state=SEED,
    )


def build_pipeline(
    model,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> Pipeline:
    """Wrap any scikit-learn model in the shared preprocessing pipeline."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])
