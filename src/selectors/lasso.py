"""
Lasso Feature Selector (Embedded).
Uses L1-penalized LogisticRegressionCV to select features with non-zero weights.
"""

import time
from typing import Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from .base import BaseFeatureSelector


class LassoFeatureSelector(BaseFeatureSelector):
    def __init__(
        self,
        cv_splits: int = 3,
        random_state: int = 42,
        max_iter: int = 1000,
        Cs: int = 10,
    ):
        super().__init__(name="Lasso")
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.Cs = Cs
        self.model_: Optional[LogisticRegressionCV] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoFeatureSelector":
        start_time = time.perf_counter()
        self.n_features_in_ = X.shape[1]

        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Use saga solver for L1 on general dimensions, or liblinear
        model = LogisticRegressionCV(
            Cs=self.Cs,
            cv=cv,
            penalty="l1",
            solver="saga",
            scoring="accuracy",
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_jobs=1,
            tol=1e-3,
        )

        model.fit(X, y)
        self.model_ = model

        # Coefficients shape: (1, n_features) or (n_classes, n_features)
        coef = model.coef_
        if coef.ndim > 1:
            # Non-zero in any class
            active_mask = np.any(np.abs(coef) > 1e-6, axis=0)
        else:
            active_mask = np.abs(coef) > 1e-6

        # Safety fallback: if all shrunk to zero, pick top feature with largest magnitude
        if not active_mask.any():
            max_idx = np.argmax(np.max(np.abs(coef), axis=0))
            active_mask[max_idx] = True

        self.support_mask_ = active_mask
        self.selected_indices_ = np.where(self.support_mask_)[0]
        self.coef_ = coef
        self.runtime_ = time.perf_counter() - start_time
        return self
