"""
Base interface for Feature Selectors.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BaseFeatureSelector(ABC):
    def __init__(self, name: str):
        self.name = name
        self.selected_indices_: Optional[np.ndarray] = None
        self.support_mask_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.runtime_: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseFeatureSelector":
        """Fit feature selector on training data."""
        pass

    def get_support(self) -> np.ndarray:
        """Returns boolean mask of selected features."""
        if self.support_mask_ is None:
            raise RuntimeError("Selector has not been fitted yet.")
        return self.support_mask_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Selects features from input array X."""
        mask = self.get_support()
        return X[:, mask]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def n_selected(self) -> int:
        return int(np.sum(self.get_support()))

    @property
    def reduction_rate(self) -> float:
        if self.n_features_in_ == 0:
            return 0.0
        return 1.0 - (self.n_selected / float(self.n_features_in_))
