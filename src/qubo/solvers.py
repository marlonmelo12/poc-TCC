"""
QUBO Solvers module: Simulated Annealing (SA) and Simulated Bifurcation (SB).
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np

try:
    import dimod
    import neal
except ImportError:
    dimod = None
    neal = None

try:
    import torch
    import simulated_bifurcation as sb
except ImportError:
    sb = None
    torch = None

from .formulation import build_qubo_dict, build_qubo_matrix_symmetric, compute_energy_analytical


def solve_qubo_sa(
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
    num_sweeps: int = 1000,
    num_reads: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Solves QUBO using D-Wave Neal SimulatedAnnealingSampler.
    Returns:
        best_solution: binary array of shape (M,)
        best_energy: analytical energy of best solution
        runtime: runtime in seconds
        info: dictionary of execution metadata
    """
    if neal is None or dimod is None:
        raise ImportError("Packages 'dimod' and 'dwave-neal' are required for Simulated Annealing solver.")

    qubo_dict = build_qubo_dict(r, d, K, alpha=alpha, beta=beta, lambda_=lambda_)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

    sampler = neal.SimulatedAnnealingSampler()

    start_time = time.perf_counter()
    response = sampler.sample(
        bqm,
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        seed=seed,
    )
    runtime = time.perf_counter() - start_time

    best_sample = response.first.sample
    n_features = len(r)
    best_solution = np.array([best_sample.get(i, 0) for i in range(n_features)], dtype=np.int64)

    best_energy = compute_energy_analytical(best_solution, r, d, K, alpha, beta, lambda_)

    info = {
        "solver": "QUBO-SA",
        "num_sweeps": num_sweeps,
        "num_reads": num_reads,
        "seed": seed,
        "raw_energy": float(response.first.energy),
        "selected_k": int(np.sum(best_solution)),
        "target_k": K,
        "cardinality_satisfied": bool(np.sum(best_solution) == K),
    }

    return best_solution, best_energy, runtime, info


def solve_qubo_sb(
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
    time_step: float = 0.1,
    steps: int = 2000,
    agents: int = 256,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Solves QUBO using Toshiba Simulated Bifurcation (SB).
    Decomposes into zero-diagonal quadratic tensor and explicit linear vector
    under discrete SB mode to prevent spurious self-coupling and non-convergence.
    Returns:
        best_solution: binary array of shape (M,)
        best_energy: analytical energy of best solution
        runtime: runtime in seconds
        info: dictionary of execution metadata
    """
    if sb is None:
        raise ImportError("Package 'simulated-bifurcation' is required for Simulated Bifurcation solver.")

    start_time = time.perf_counter()

    if seed is not None and torch is not None:
        torch.manual_seed(seed)

    use_cuda = torch is not None and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Pure quadratic matrix with zero diagonal (prevents non-zero J_ii self-coupling in Ising)
    Q_quad = 0.5 * beta * d + lambda_
    np.fill_diagonal(Q_quad, 0.0)

    # Explicit linear vector
    l_vec = -alpha * r + lambda_ * (1.0 - 2.0 * K)

    Q_tensor = torch.tensor(Q_quad, dtype=torch.float32, device=device)
    l_tensor = torch.tensor(l_vec, dtype=torch.float32, device=device)

    best_solution_tensor, sb_energy = sb.minimize(
        Q_tensor,
        l_tensor,
        domain="binary",
        agents=agents,
        max_steps=steps,
        mode="discrete",
        early_stopping=False,
        best_only=True,
        verbose=False,
    )
    runtime = time.perf_counter() - start_time

    if torch is not None and isinstance(best_solution_tensor, torch.Tensor):
        best_solution = best_solution_tensor.detach().cpu().numpy().flatten().astype(np.int64)
    else:
        best_solution = np.asarray(best_solution_tensor, dtype=np.int64).flatten()

    # Safety safeguard: if solver collapsed to all zeros, activate top K by relevance
    fallback_used = False
    if np.sum(best_solution) == 0:
        top_k = np.argsort(r)[::-1][:K]
        best_solution = np.zeros(len(r), dtype=np.int64)
        best_solution[top_k] = 1
        fallback_used = True

    best_energy = compute_energy_analytical(best_solution, r, d, K, alpha, beta, lambda_)

    info = {
        "solver": "QUBO-SB",
        "max_steps": steps,
        "agents": agents,
        "seed": seed,
        "selected_k": int(np.sum(best_solution)),
        "target_k": K,
        "cardinality_satisfied": bool(np.sum(best_solution) == K),
        "fallback_used": fallback_used,
    }

    return best_solution, best_energy, runtime, info



def solve_qubo(
    solver_name: str,
    r: np.ndarray,
    d: np.ndarray,
    K: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 2.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Unified entry point for QUBO solvers.
    """
    solver_clean = solver_name.strip().lower()
    if "sa" in solver_clean or "anneal" in solver_clean:
        num_sweeps = kwargs.get("num_sweeps", 1000)
        num_reads = kwargs.get("num_reads", 10)
        return solve_qubo_sa(
            r, d, K, alpha=alpha, beta=beta, lambda_=lambda_,
            num_sweeps=num_sweeps, num_reads=num_reads, seed=seed
        )
    elif "sb" in solver_clean or "bifurcation" in solver_clean:
        steps = kwargs.get("steps", 1000)
        agents = kwargs.get("agents", 256)
        return solve_qubo_sb(
            r, d, K, alpha=alpha, beta=beta, lambda_=lambda_,
            steps=steps, agents=agents, seed=seed
        )
    else:
        raise ValueError(f"Unknown QUBO solver: {solver_name}")
