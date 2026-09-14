"""
ANOVA F-score Feature Selector (Filter).
Uses sklearn f_classif to compute F-scores and retain top percentile or top K features.
"""

import time
from typing import Optional
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile
from .base import BaseFeatureSelector


class ANOVAFeatureSelector(BaseFeatureSelector):
    def __init__(self, k: Optional[int] = None, percentile: Optional[float] = 10.0):
        super().__init__(name="ANOVA")
        self.k = k
        self.percentile = percentile
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANOVAFeatureSelector":
        start_time = time.perf_counter()
        self.n_features_in_ = X.shape[1]

        scores, pvalues = f_classif(X, y)
        # Handle NaNs if features are constant
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = scores
        self.pvalues_ = pvalues

        if self.k is not None:
            # Select top K
            actual_k = min(self.k, self.n_features_in_)
            selector = SelectKBest(score_func=lambda X, y: (scores, pvalues), k=actual_k)
            selector.fit(X, y)
            self.support_mask_ = selector.get_support()
        else:
            pct = self.percentile if self.percentile is not None else 10.0
            selector = SelectPercentile(score_func=lambda X, y: (scores, pvalues), percentile=pct)
            selector.fit(X, y)
            self.support_mask_ = selector.get_support()
            # Ensure at least 1 feature is selected
            if not self.support_mask_.any():
                top_1 = np.argmax(scores)
                self.support_mask_[top_1] = True

        self.selected_indices_ = np.where(self.support_mask_)[0]
        self.runtime_ = time.perf_counter() - start_time
        return self
