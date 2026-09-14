"""
Recursive Feature Elimination with Cross-Validation (RFECV) Selector (Wrapper).
Optimized implementation:
- Estimator: RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
- step: 0.10 (prunes 10% per round)
- cv: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
Executes strictly on training data without test data leakage.
"""

import time
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from .base import BaseFeatureSelector


class RFECVFeatureSelector(BaseFeatureSelector):
    def __init__(
        self,
        n_estimators: int = 50,
        step: float = 0.10,
        cv_splits: int = 3,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        super().__init__(name="RFECV")
        self.n_estimators = n_estimators
        self.step = step
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ranking_: Optional[np.ndarray] = None
        self.grid_scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFECVFeatureSelector":
        start_time = time.perf_counter()
        self.n_features_in_ = X.shape[1]

        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        selector = RFECV(
            estimator=estimator,
            step=self.step,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,  # RF estimator handles multi-threading via n_jobs
        )

        selector.fit(X, y)

        self.support_mask_ = selector.support_
        self.selected_indices_ = np.where(self.support_mask_)[0]
        self.ranking_ = selector.ranking_
        self.runtime_ = time.perf_counter() - start_time
        return self
