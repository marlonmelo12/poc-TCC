"""
QUBO Feature Selector: Wraps QUBO formulation with SA or SB solvers.
"""

import time
from typing import Optional, Tuple, Dict, Any
import numpy as np

from .base import BaseFeatureSelector
from ..qubo.metrics_mrmr import compute_mrmr_matrices
from ..qubo.formulation import calibrate_lambda
from ..qubo.solvers import solve_qubo


class QUBOFeatureSelector(BaseFeatureSelector):
    def __init__(
        self,
        solver_name: str = "QUBO-SA",
        K: int = 10,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_: Optional[float] = None,
        seed: int = 42,
        precomputed_mrmr: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **solver_kwargs,
    ):
        super().__init__(name=solver_name)
        self.solver_name = solver_name
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.seed = seed
        self.precomputed_mrmr = precomputed_mrmr
        self.solver_kwargs = solver_kwargs

        self.r_: Optional[np.ndarray] = None
        self.d_: Optional[np.ndarray] = None
        self.qubo_energy_: Optional[float] = None
        self.solver_info_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QUBOFeatureSelector":
        start_time = time.perf_counter()
        self.n_features_in_ = X.shape[1]

        # Use precomputed mRMR matrices if available for this fold
        if self.precomputed_mrmr is not None:
            r, d = self.precomputed_mrmr
        else:
            r, d = compute_mrmr_matrices(X, y, random_state=self.seed)

        self.r_ = r
        self.d_ = d

        # Calibrate lambda if not specified
        lambda_val = self.lambda_
        if lambda_val is None:
            lambda_val = calibrate_lambda(r, beta=self.beta, K=self.K)

        best_solution, best_energy, solver_time, info = solve_qubo(
            solver_name=self.solver_name,
            r=r,
            d=d,
            K=self.K,
            alpha=self.alpha,
            beta=self.beta,
            lambda_=lambda_val,
            seed=self.seed,
            **self.solver_kwargs,
        )

        # Safety safeguard: prevent zero features from reaching classifiers
        if np.sum(best_solution) == 0:
            top_k_indices = np.argsort(r)[::-1][:self.K]
            best_solution = np.zeros(self.n_features_in_, dtype=np.int64)
            best_solution[top_k_indices] = 1
            if info is not None:
                info["fallback_used"] = True

        self.support_mask_ = (best_solution == 1)
        self.selected_indices_ = np.where(self.support_mask_)[0]
        self.qubo_energy_ = best_energy
        self.solver_info_ = info
        self.runtime_ = time.perf_counter() - start_time
        return self
