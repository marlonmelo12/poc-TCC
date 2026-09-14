"""
QUBO formulation for feature selection inspired by mRMR.
Formulation:
    E(x) = - alpha * sum_i(r_i * x_i)
           + beta * sum_{i < j}(d_ij * x_i * x_j)
           + lambda * (sum_i(x_i) - K)^2
"""

from typing import Dict, Tuple, Union
import numpy as np

from .metrics_mrmr import compute_redundancy_spearman, compute_relevance

# Backward-compatibility alias
compute_spearman_matrix_fast = compute_redundancy_spearman


def compute_energy_analytical(
    x: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
) -> float:
    """
    Computes exact analytical energy of a binary vector x according to the objective function:
        E(x) = - alpha * sum_i(r_i * x_i)
               + beta * sum_{i < j}(d_ij * x_i * x_j)
               + lambda * (sum_i(x_i) - K)^2
    """
    x = np.asarray(x, dtype=np.float64)
    relevance_term = -alpha * np.sum(r * x)

    # Quadratic redundancy: sum_{i < j} d_ij * x_i * x_j
    # Since d has 0 on diagonal, (x.T @ d @ x) = 2 * sum_{i < j} d_ij x_i x_j
    redundancy_term = beta * 0.5 * float(x.T @ (d @ x))

    cardinality_term = lambda_ * float((np.sum(x) - K) ** 2)

    return float(relevance_term + redundancy_term + cardinality_term)


def build_qubo_matrix_symmetric(
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
) -> np.ndarray:
    """
    Constructs the dense symmetric QUBO matrix Q_sym such that:
        x.T @ Q_sym @ x + lambda * K^2 = E(x)

    Mathematical Derivation:
        In continuous/discrete quadratic form x.T @ Q_sym @ x:
            x.T @ Q_sym @ x = sum_i Q_ii x_i^2 + sum_{i != j} Q_ij x_i x_j
                            = sum_i Q_ii x_i   + 2 * sum_{i < j} Q_ij x_i x_j   (since Q is symmetric)
        Equating to the analytical objective:
            E(x) = sum_i (-alpha * r_i + lambda * (1 - 2*K)) x_i
                 + sum_{i < j} (beta * d_ij + 2 * lambda) x_i x_j
                 + lambda * K^2
        Therefore:
            Diagonal elements (i == j):
                Q_ii = - alpha * r_i + lambda * (1 - 2*K)
            Off-diagonal elements (i != j):
                2 * Q_ij = beta * d_ij + 2 * lambda  ==>  Q_ij = 0.5 * beta * d_ij + lambda
    """
    n_features = len(r)
    Q = np.empty((n_features, n_features), dtype=np.float64)

    # Off-diagonal: symmetric (d has 0 on diagonal)
    Q[:] = 0.5 * beta * d + lambda_

    # Diagonal elements:
    np.fill_diagonal(Q, -alpha * r + lambda_ * (1.0 - 2.0 * K))

    return Q


def matrix_to_qubo_dict(Q_sym: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Converts a dense symmetric QUBO matrix into D-Wave dimod upper-triangular dictionary format.
    Folds off-diagonal pairs (i, j) and (j, i) into a single upper-triangular entry (i < j):
        Q_dict[(i, j)] = Q_sym[i, j] + Q_sym[j, i] = 2 * Q_sym[i, j]
    Preserves exact energy:
        dimod.BinaryQuadraticModel.from_qubo(Q_dict).energy(x) == x.T @ Q_sym @ x
    """
    n_features = Q_sym.shape[0]
    qubo = {}

    for i in range(n_features):
        qubo[(i, i)] = float(Q_sym[i, i])
        for j in range(i + 1, n_features):
            val = float(Q_sym[i, j] + Q_sym[j, i])
            if val != 0.0:
                qubo[(i, j)] = val

    return qubo


def qubo_dict_to_matrix_symmetric(
    qubo_dict: Dict[Tuple[int, int], float], n_features: int
) -> np.ndarray:
    """
    Converts an upper-triangular QUBO dictionary into a dense symmetric matrix.
    Splits off-diagonal entries equally across (i, j) and (j, i):
        Q_sym[i, j] = Q_sym[j, i] = 0.5 * Q_dict[(i, j)]
    """
    Q = np.zeros((n_features, n_features), dtype=np.float64)
    for (i, j), val in qubo_dict.items():
        if i == j:
            Q[i, i] = float(val)
        else:
            half = 0.5 * float(val)
            Q[i, j] = half
            Q[j, i] = half
    return Q


def build_qubo_dict(
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
) -> Dict[Tuple[int, int], float]:
    """
    Constructs upper triangular QUBO dictionary format {(i, j): bias} for D-Wave dimod.
    Derived canonically from build_qubo_matrix_symmetric to ensure Single Source of Truth:
        Q_dict[(i, i)] = - alpha * r_i + lambda * (1 - 2*K)
        Q_dict[(i, j)] = beta * d_ij + 2 * lambda   (for i < j)
    """
    Q_sym = build_qubo_matrix_symmetric(r, d, K, alpha=alpha, beta=beta, lambda_=lambda_)
    return matrix_to_qubo_dict(Q_sym)


def calibrate_lambda(
    r: np.ndarray,
    beta: float = 1.0,
    K: int = 10,
    alpha: float = 1.0,
) -> float:
    """
    Dynamic cardinality penalty parameter lambda(K):
    Scales sublinearly with K to balance cumulative pairwise redundancy
    against the quadratic cardinality constraint without inducing numerical stiffness
    or freezing in Simulated Bifurcation Kerr oscillators.
    """
    max_relevance = float(np.max(r)) if len(r) > 0 else 1.0
    base_penalty = alpha * max_relevance
    # Redundancy grows with selected set; sqrt(K) + linear scaling ensures sufficient constraint pressure
    k_factor = 0.5 * beta * np.sqrt(max(1, K)) + 0.1 * beta * float(K)
    lambda_val = max(2.5, base_penalty + k_factor)
    return float(lambda_val)


def project_cardinality_qubo(
    x: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Exact discrete projection of a binary solution x onto the cardinality hyperplane sum(x) = K.
    Uses the exact marginal gradient of the unconstrained mRMR objective:
        grad_i E(x) = -alpha * r_i + beta * (d @ x)_i
    - If sum(x) < K: selects the (K - sum(x)) unselected features with lowest marginal cost.
    - If sum(x) > K: removes the (sum(x) - K) selected features with highest marginal cost.
    - If sum(x) == K: returns unmodified copy.
    Runtime complexity: O(p) via BLAS matrix-vector product.
    """
    x_proj = np.asarray(x, dtype=np.int64).copy()
    k_curr = int(np.sum(x_proj))
    if k_curr == K:
        return x_proj

    n_features = len(x_proj)
    K_target = min(max(1, K), n_features)

    # Marginal gradient: cost of each feature given currently selected set
    # grad_i = -alpha * r_i + beta * sum_{j in selected} d_ij
    marginal_cost = -alpha * r + beta * (d @ x_proj)

    if k_curr < K_target:
        n_to_add = K_target - k_curr
        # Mask out already selected features so they cannot be picked again
        marginal_cost[x_proj == 1] = np.inf
        # Pick n_to_add features with lowest marginal cost
        best_indices = np.argpartition(marginal_cost, n_to_add)[:n_to_add]
        x_proj[best_indices] = 1
    elif k_curr > K_target:
        n_to_remove = k_curr - K_target
        # Mask out unselected features so only currently selected can be pruned
        marginal_cost[x_proj == 0] = -np.inf
        # Pick n_to_remove features with highest marginal cost (worst contribution)
        worst_indices = np.argpartition(marginal_cost, -n_to_remove)[-n_to_remove:]
        x_proj[worst_indices] = 0

    return x_proj
