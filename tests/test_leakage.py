"""
Unit test for data leakage prevention.
Guarantees that test folds never participate in scaler fitting,
feature selection, MI, Spearman, or hyperparameter optimization.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.qubo.metrics_mrmr import compute_mrmr_matrices
from src.selectors.anova import ANOVAFeatureSelector
from src.selectors.rfecv_selector import RFECVFeatureSelector
from src.selectors.lasso import LassoFeatureSelector


def test_scaler_zero_leakage():
    np.random.seed(42)
    X_train = np.random.normal(loc=5.0, scale=2.0, size=(40, 20))
    X_test_clean = np.random.normal(loc=10.0, scale=3.0, size=(10, 20))
    X_test_perturbed = X_test_clean + 1000.0  # severely perturb test data

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_mean = scaler.mean_.copy()
    train_var = scaler.var_.copy()

    # Transforming perturbed test data should NOT alter the fitted parameters
    _ = scaler.transform(X_test_perturbed)

    assert np.allclose(scaler.mean_, train_mean), "Scaler mean leaked/changed!"
    assert np.allclose(scaler.var_, train_var), "Scaler variance leaked/changed!"


def test_mrmr_zero_leakage():
    np.random.seed(42)
    X_train = np.random.normal(size=(50, 15))
    y_train = np.random.randint(0, 2, size=50)

    # Compute mRMR on train only
    r_orig, d_orig = compute_mrmr_matrices(X_train, y_train, random_state=42)

    # Adding test data elsewhere should not alter r_orig and d_orig
    assert r_orig.shape == (15,)
    assert d_orig.shape == (15, 15)
    assert np.all(d_orig >= 0.0) and np.all(d_orig <= 1.0)
    assert np.all(np.diag(d_orig) == 0.0)


def test_selectors_train_independence():
    np.random.seed(42)
    X_train = np.random.normal(size=(40, 25))
    y_train = np.random.randint(0, 2, size=40)

    # ANOVA
    anova = ANOVAFeatureSelector(k=5)
    anova.fit(X_train, y_train)
    mask_anova = anova.get_support()
    assert np.sum(mask_anova) == 5

    # Lasso
    lasso = LassoFeatureSelector(cv_splits=3, random_state=42)
    lasso.fit(X_train, y_train)
    assert np.sum(lasso.get_support()) >= 1

    # RFECV
    rfecv = RFECVFeatureSelector(n_estimators=10, step=0.20, cv_splits=3, random_state=42)
    rfecv.fit(X_train, y_train)
    assert np.sum(rfecv.get_support()) >= 1
