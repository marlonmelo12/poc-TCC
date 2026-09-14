"""
mRMR metrics computation: Mutual Information (Relevance) and Absolute Spearman Correlation (Redundancy).
Optimized with BLAS vectorization for high-dimensional feature spaces.
"""

from typing import Tuple
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif


def compute_relevance(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Computes Mutual Information (MI) between each feature and the target variable,
    strictly on training data, normalized via MinMax scaling to [0, 1].
    """
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    mi_min = np.min(mi)
    mi_max = np.max(mi)

    if mi_max > mi_min:
        r_norm = (mi - mi_min) / (mi_max - mi_min)
    else:
        r_norm = np.ones_like(mi)

    return r_norm.astype(np.float64)


def compute_redundancy_spearman(X_train: np.ndarray) -> np.ndarray:
    """
    Computes pairwise absolute Spearman rank correlation matrix strictly on training data.
    Uses vectorized rank standardization and BLAS matrix multiplication:
        rho = (1 / (N - 1)) * Z_R.T @ Z_R
    Time complexity: O(N * M + N * M^2) via BLAS dgemm, taking <0.5s for M=6033.
    Returns:
        M x M matrix of absolute Spearman correlations with diagonal set to 0.
    """
    n_samples, n_features = X_train.shape

    # Rank columns
    # scipy.stats.rankdata along axis=0 ranks each column independently
    ranks = rankdata(X_train, axis=0)

    # Standardize ranked columns
    mean_ranks = np.mean(ranks, axis=0, keepdims=True)
    std_ranks = np.std(ranks, axis=0, ddof=1, keepdims=True)
    # Replace zero std with 1.0 to prevent division by zero for constant features
    std_ranks[std_ranks == 0.0] = 1.0

    z_ranks = (ranks - mean_ranks) / std_ranks

    # Pairwise correlation via dot product
    corr_matrix = (z_ranks.T @ z_ranks) / (n_samples - 1.0)

    # Numerical clipping to [-1, 1]
    np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)

    # Absolute value
    d_matrix = np.abs(corr_matrix)

    # Zero out diagonal
    np.fill_diagonal(d_matrix, 0.0)

    return d_matrix.astype(np.float64)


def compute_mrmr_matrices(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both relevance vector r in [0, 1]^M and redundancy matrix d in [0, 1]^{M x M}.
    """
    r = compute_relevance(X_train, y_train, random_state=random_state)
    d = compute_redundancy_spearman(X_train)
    return r, d
