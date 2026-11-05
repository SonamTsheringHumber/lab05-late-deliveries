"""Metrics, the cost rule and the cost-based threshold choice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

VOUCHER_COST = 4
LATE_COST = 15
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.05, 0.95 + 1e-9, 0.05)]

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_cv(seed: int, n_splits: int = 5) -> StratifiedKFold:
    """The one cross-validation splitter used for every comparison."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def order_cost(y_true, flagged) -> int:
    """$4 for every flagged order plus $15 for every late order not flagged."""
    late = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    return int(VOUCHER_COST * flagged.sum() + LATE_COST * (late & ~flagged).sum())


def cost_by_threshold(y_true, proba, thresholds=THRESHOLDS) -> dict[float, int]:
    """Total cost at each threshold. An order is flagged when proba >= threshold."""
    proba = np.asarray(proba)
    return {t: order_cost(y_true, proba >= t) for t in thresholds}


def choose_threshold(y_true, proba, thresholds=THRESHOLDS) -> float:
    """Threshold with the lowest total cost (the smallest one wins a tie)."""
    costs = cost_by_threshold(y_true, proba, thresholds)
    return min(costs, key=lambda t: (costs[t], t))


def policy_costs(y_true, proba, threshold) -> dict[str, int]:
    """Cost of flagging nobody, flagging everybody, and using the model."""
    y_true = np.asarray(y_true)
    return {
        "flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "model": order_cost(y_true, np.asarray(proba) >= threshold),
    }


def final_metrics(y_true, proba, threshold) -> dict:
    """Confusion matrix and scores at a threshold (for the one test-set look)."""
    flagged = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def compare_models(models: dict, X, y, seed: int) -> pd.DataFrame:
    """Five-fold stratified CV on training data. Mean and std of each metric."""
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=make_cv(seed), scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)
