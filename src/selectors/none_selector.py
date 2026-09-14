"""
None selector: Baseline without feature selection (all features retained).
"""

import time
import numpy as np
from .base import BaseFeatureSelector


class NoneFeatureSelector(BaseFeatureSelector):
    def __init__(self):
        super().__init__(name="None")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoneFeatureSelector":
        start_time = time.perf_counter()
        self.n_features_in_ = X.shape[1]
        self.support_mask_ = np.ones(self.n_features_in_, dtype=bool)
        self.selected_indices_ = np.arange(self.n_features_in_)
        self.runtime_ = time.perf_counter() - start_time
        return self
