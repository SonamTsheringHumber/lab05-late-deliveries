"""Run the whole project: python -m latedeliveries.train."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    THRESHOLDS,
    choose_threshold,
    compare_models,
    cost_by_threshold,
    final_metrics,
    make_cv,
    policy_costs,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    split_orders,
)

ROOT = Path(__file__).resolve().parents[2]

# Chosen after reading reports/cv_results.csv; the reasons go in the README.
CHOSEN_MODEL = "logistic_regression"

PARAM_GRID = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"],
}


def candidate_models() -> dict:
    """The baseline and the real candidates, all wrapped in the same pipeline."""
    models = {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=SEED
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }
    return {name: build_pipeline(model) for name, model in models.items()}


def leakage_check(X_train, y_train) -> dict:
    """ROC-AUC with and without the after-delivery column customer_rating."""
    rating = make_orders().loc[X_train.index, "customer_rating"]
    X_leaky = X_train.assign(customer_rating=rating)
    model = LogisticRegression(max_iter=1000)
    honest = build_pipeline(model)
    leaky = build_pipeline(
        model,
        numeric=[*NUMERIC_FEATURES, "customer_rating"],
        categorical=CATEGORICAL_FEATURES,
    )
    cv = make_cv(SEED)
    auc_without = cross_val_score(
        honest, X_train, y_train, cv=cv, scoring="roc_auc"
    ).mean()
    auc_with = cross_val_score(leaky, X_leaky, y_train, cv=cv, scoring="roc_auc").mean()
    return {
        "roc_auc_without_customer_rating": float(auc_without),
        "roc_auc_with_customer_rating": float(auc_with),
        "difference": float(auc_with - auc_without),
    }


def main(root: Path = ROOT) -> dict:
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # 1. Split. The test set is not touched again until step 5.
    X_train, X_test, y_train, y_test = split_orders()
    cv = make_cv(SEED)

    # 2. Compare the baseline and the candidates, training data only.
    pipelines = candidate_models()
    results = compare_models(pipelines, X_train, y_train, SEED)
    best_threshold = {}
    best_cost = {}
    for name, pipe in pipelines.items():
        proba = cross_val_predict(
            pipe, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        best = choose_threshold(y_train, proba)
        best_threshold[name] = best
        best_cost[name] = cost_by_threshold(y_train, proba)[best]
    results["cv_best_threshold"] = results["model"].map(best_threshold)
    results["cv_best_cost"] = results["model"].map(best_cost)
    results.round(4).to_csv(reports_dir / "cv_results.csv", index=False)

    # 3. Tune the chosen model with a grid search, training data only.
    untuned_auc = cross_val_score(
        pipelines[CHOSEN_MODEL], X_train, y_train, cv=cv, scoring="roc_auc"
    )
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
    )
    search.fit(X_train, y_train)
    i = search.best_index_
    tuning = {
        "scoring": "roc_auc",
        "n_combinations": len(search.cv_results_["params"]),
        "best_params": dict(search.best_params_),
        "best_cv_roc_auc": float(search.best_score_),
        "best_cv_roc_auc_std": float(search.cv_results_["std_test_score"][i]),
        "untuned_cv_roc_auc": float(untuned_auc.mean()),
        "untuned_cv_roc_auc_std": float(untuned_auc.std()),
    }
    gain = tuning["best_cv_roc_auc"] - tuning["untuned_cv_roc_auc"]
    tuning["gain_over_untuned"] = float(gain)
    tuning["gain_larger_than_fold_spread"] = bool(gain > tuning["best_cv_roc_auc_std"])

    # 4. Threshold from out-of-fold probabilities, training data only.
    proba_oof = cross_val_predict(
        search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, proba_oof)
    curve = cost_by_threshold(y_train, proba_oof)
    pd.DataFrame({"threshold": list(curve), "cost": list(curve.values())}).to_csv(
        reports_dir / "threshold_costs.csv", index=False
    )

    # 5. Fit on the whole training set and score the test set once.
    final = search.best_estimator_
    final.fit(X_train, y_train)
    proba_test = final.predict_proba(X_test)[:, 1]
    metrics = final_metrics(y_test, proba_test, threshold)
    metrics["policy_costs"] = policy_costs(y_test, proba_test, threshold)
    metrics["n_test_orders"] = int(len(y_test))
    metrics["n_late_in_test"] = int(y_test.sum())
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Extra: see leakage for yourself (training data only).
    (reports_dir / "leakage_check.json").write_text(
        json.dumps(leakage_check(X_train, y_train), indent=2) + "\n"
    )

    # 6. Save the pipeline and its information.
    joblib.dump(final, models_dir / "late_orders.joblib")
    info = {
        "model": CHOSEN_MODEL,
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "data_seed": 7,
        "threshold": threshold,
        "threshold_grid": THRESHOLDS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "best_settings": tuning["best_params"],
        "tuning": tuning,
        "test_results": metrics,
    }
    (models_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["test_results"], indent=2))
