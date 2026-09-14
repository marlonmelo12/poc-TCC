"""
Unit tests for classifier implementations and HPO fitting:
- Verifies RF, KNN, SVM, XGBoost, CatBoost, OPF.
- Guarantees zero FutureWarning for SVM (using CalibratedClassifierCV).
- Verifies predict_proba and probability bounds.
"""

import warnings
import numpy as np
import pytest
from src.models.classifiers import fit_classifier_with_hpo, SimpleOPFClassifier
from src.pipeline.evaluation import evaluate_predictions


def test_all_classifiers_hpo_and_prediction():
    np.random.seed(42)
    X_train = np.random.randn(40, 15)
    y_train = np.random.randint(0, 2, size=40)
    X_test = np.random.randn(10, 15)
    y_test = np.random.randint(0, 2, size=10)

    classifiers = ["rf", "knn", "svm", "xgboost", "catboost", "opf"]

    for clf_name in classifiers:
        model, best_params = fit_classifier_with_hpo(clf_name, X_train, y_train, cv_splits=3, seed=42)
        assert model is not None
        assert isinstance(best_params, dict)

        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            assert y_prob.shape == (len(y_test), 2)
            assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)
            assert np.allclose(np.sum(y_prob, axis=1), 1.0)
        else:
            y_prob = None

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


def test_svm_calibration_no_futurewarning():
    """Verify that SVM does not emit FutureWarning for probability parameter."""
    np.random.seed(42)
    X_train = np.random.randn(30, 10)
    y_train = np.random.randint(0, 2, size=30)
    X_test = np.random.randn(10, 10)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        model, _ = fit_classifier_with_hpo("svm", X_train, y_train, cv_splits=3, seed=42)
        _ = model.predict_proba(X_test)

        # Ensure no FutureWarning about 'probability' was raised
        future_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, FutureWarning) and "probability" in str(w.message).lower()
        ]
        assert len(future_warnings) == 0, f"Unexpected FutureWarning: {future_warnings}"


def test_opf_vectorized_probabilities():
    """Verify SimpleOPFClassifier probability validity."""
    np.random.seed(42)
    X = np.random.randn(30, 20)
    y = np.random.randint(0, 2, size=30)

    opf = SimpleOPFClassifier()
    opf.fit(X, y)
    probs = opf.predict_proba(X[:5])

    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
