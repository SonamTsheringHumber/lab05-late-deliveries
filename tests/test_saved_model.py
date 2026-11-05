"""The saved pipeline must predict the same after it is loaded again."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline


def test_saved_model_predicts_the_same_after_reloading(tmp_path, split, odd_order):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    path = tmp_path / "late_orders.joblib"
    joblib.dump(pipe, path)
    reloaded = joblib.load(path)
    assert np.array_equal(pipe.predict_proba(X_test), reloaded.predict_proba(X_test))
    assert np.array_equal(
        pipe.predict_proba(odd_order), reloaded.predict_proba(odd_order)
    )
