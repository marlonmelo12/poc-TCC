"""
Unit tests for QUBO formulation, matrix symmetry, energy equivalence, and solver output.
"""

import numpy as np
import pytest
from src.qubo.formulation import (
    compute_energy_analytical,
    build_qubo_matrix_symmetric,
    build_qubo_dict,
    matrix_to_qubo_dict,
    qubo_dict_to_matrix_symmetric,
    calibrate_lambda,
)


def test_qubo_energy_matrix_equivalence():
    np.random.seed(42)
    n_features = 8
    K = 3
    alpha = 1.0
    beta = 1.0
    lambda_ = 5.0

    r = np.random.uniform(0.1, 1.0, size=n_features)
    # Symmetric redundancy matrix with 0 diagonal
    d_raw = np.random.uniform(0.0, 0.8, size=(n_features, n_features))
    d = 0.5 * (d_raw + d_raw.T)
    np.fill_diagonal(d, 0.0)

    Q_sym = build_qubo_matrix_symmetric(r, d, K, alpha=alpha, beta=beta, lambda_=lambda_)

    # Verify symmetry
    assert np.allclose(Q_sym, Q_sym.T), "QUBO matrix is not symmetric!"

    # Test across random binary configurations
    for _ in range(20):
        x = np.random.randint(0, 2, size=n_features)

        e_analytical = compute_energy_analytical(x, r, d, K, alpha, beta, lambda_)

        # Matrix energy: x.T @ Q @ x + lambda * K^2
        e_matrix = float(x.T @ Q_sym @ x + lambda_ * (K ** 2))

        assert np.isclose(e_analytical, e_matrix, atol=1e-7), (
            f"Energy mismatch: Analytical={e_analytical}, Matrix={e_matrix}"
        )


def test_qubo_cardinality_penalty():
    n_features = 6
    K = 2
    r = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1])
    d = np.zeros((n_features, n_features))
    lambda_ = 10.0

    # Optimal candidate with exactly K features
    x_valid = np.array([1, 1, 0, 0, 0, 0])
    e_valid = compute_energy_analytical(x_valid, r, d, K, lambda_=lambda_)

    # Candidate with 3 features (violates K)
    x_invalid_3 = np.array([1, 1, 1, 0, 0, 0])
    e_invalid_3 = compute_energy_analytical(x_invalid_3, r, d, K, lambda_=lambda_)

    # Candidate with 1 feature (violates K)
    x_invalid_1 = np.array([1, 0, 0, 0, 0, 0])
    e_invalid_1 = compute_energy_analytical(x_invalid_1, r, d, K, lambda_=lambda_)

    # The penalty should make invalid configurations strictly higher energy
    assert e_valid < e_invalid_3, "Cardinality penalty failed to penalize extra feature!"
    assert e_valid < e_invalid_1, "Cardinality penalty failed to penalize missing feature!"


def test_qubo_dict_format():
    n_features = 4
    K = 2
    r = np.array([0.5, 0.6, 0.7, 0.8])
    d = np.array([
        [0.0, 0.2, 0.3, 0.4],
        [0.2, 0.0, 0.1, 0.5],
        [0.3, 0.1, 0.0, 0.2],
        [0.4, 0.5, 0.2, 0.0],
    ])
    qubo_dict = build_qubo_dict(r, d, K, alpha=1.0, beta=1.0, lambda_=2.0)

    # Check upper triangular keys
    for (i, j) in qubo_dict.keys():
        assert i <= j, f"Key ({i}, {j}) is not upper triangular!"


def test_sa_solver():
    from src.qubo.solvers import solve_qubo_sa
    n_features = 10
    K = 3
    r = np.linspace(0.1, 1.0, n_features)
    d = np.zeros((n_features, n_features))
    np.fill_diagonal(d, 0.0)

    sol, energy, runtime, info = solve_qubo_sa(r, d, K=K, alpha=1.0, beta=0.0, lambda_=2.0, num_sweeps=1000, num_reads=10, seed=42)

    assert len(sol) == n_features
    assert np.all(np.isin(sol, [0, 1]))
    assert np.sum(sol) == K, f"SA failed to satisfy cardinality constraint K={K}: sum={np.sum(sol)}"
    # Since beta=0 and r is increasing, optimal solution must pick the top 3 features (indices 7, 8, 9)
    assert np.all(sol[-3:] == 1), f"SA failed to pick top features: {sol}"


def test_sb_solver():
    from src.qubo.solvers import solve_qubo_sb
    n_features = 10
    K = 3
    r = np.linspace(0.1, 1.0, n_features)
    d = np.zeros((n_features, n_features))
    np.fill_diagonal(d, 0.0)

    sol, energy, runtime, info = solve_qubo_sb(r, d, K=K, alpha=1.0, beta=0.0, lambda_=2.0, steps=1000, seed=42)

    assert len(sol) == n_features
    assert np.all(np.isin(sol, [0, 1]))
    assert np.sum(sol) == K, f"SB failed to satisfy cardinality constraint K={K}: sum={np.sum(sol)}"
    assert np.all(sol[-3:] == 1), f"SB failed to pick top features: {sol}"


def test_qubo_dimod_energy_equivalence():
    """
    Validates that D-Wave dimod BQM evaluates to the exact same energy as
    the analytical function and the symmetric matrix across random configurations.
    """
    try:
        import dimod
    except ImportError:
        pytest.skip("dimod not installed")

    np.random.seed(42)
    n_features = 6
    K = 2
    alpha = 1.0
    beta = 1.0
    lambda_ = 3.0

    r = np.random.uniform(0.1, 1.0, size=n_features)
    d_raw = np.random.uniform(0.0, 0.5, size=(n_features, n_features))
    d = 0.5 * (d_raw + d_raw.T)
    np.fill_diagonal(d, 0.0)

    Q_sym = build_qubo_matrix_symmetric(r, d, K, alpha=alpha, beta=beta, lambda_=lambda_)
    q_dict = build_qubo_dict(r, d, K, alpha=alpha, beta=beta, lambda_=lambda_)
    bqm = dimod.BinaryQuadraticModel.from_qubo(q_dict)

    for _ in range(25):
        x = np.random.randint(0, 2, size=n_features)
        sample = {i: int(x[i]) for i in range(n_features)}

        e_analytical = compute_energy_analytical(x, r, d, K, alpha, beta, lambda_)
        e_matrix = float(x.T @ Q_sym @ x + lambda_ * (K ** 2))
        e_dimod = float(bqm.energy(sample) + lambda_ * (K ** 2))

        assert np.isclose(e_analytical, e_matrix, atol=1e-7)
        assert np.isclose(e_analytical, e_dimod, atol=1e-7)


def test_qubo_matrix_dict_roundtrip():
    """
    Validates exact roundtrip conversion between dense symmetric matrix and
    upper-triangular QUBO dictionary format.
    """
    np.random.seed(123)
    n = 8
    A = np.random.uniform(-5.0, 5.0, (n, n))
    Q_sym = 0.5 * (A + A.T)

    qubo_dict = matrix_to_qubo_dict(Q_sym)
    Q_recovered = qubo_dict_to_matrix_symmetric(qubo_dict, n)

    assert np.allclose(Q_sym, Q_recovered, atol=1e-12)


def test_calibrate_lambda_dynamic_scaling():
    """Verify that calibrate_lambda scales dynamically with K to maintain constraint pressure."""
    r = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    lambda_10 = calibrate_lambda(r, beta=1.0, K=10)
    lambda_20 = calibrate_lambda(r, beta=1.0, K=20)
    lambda_50 = calibrate_lambda(r, beta=1.0, K=50)
    lambda_100 = calibrate_lambda(r, beta=1.0, K=100)

    assert lambda_10 >= 2.5
    assert lambda_10 < lambda_20 < lambda_50 < lambda_100, (
        f"Lambda must scale monotonically with K: {[lambda_10, lambda_20, lambda_50, lambda_100]}"
    )


def test_project_cardinality_qubo():
    """Verify exact discrete projection on the cardinality simplex."""
    from src.qubo.formulation import project_cardinality_qubo

    np.random.seed(42)
    n = 50
    r = np.linspace(0.1, 1.0, n)
    d = np.random.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(d, 0.0)

    # 1. Under-cardinality: x has 5 features, target is 15
    x_under = np.zeros(n, dtype=np.int64)
    x_under[:5] = 1
    x_proj_under = project_cardinality_qubo(x_under, r, d, K=15)
    assert np.sum(x_proj_under) == 15
    # Original 5 should remain selected
    assert np.all(x_proj_under[:5] == 1)

    # 2. Over-cardinality: x has 25 features, target is 10
    x_over = np.zeros(n, dtype=np.int64)
    x_over[:25] = 1
    x_proj_over = project_cardinality_qubo(x_over, r, d, K=10)
    assert np.sum(x_proj_over) == 10
    # Selected must be subset of original 25
    assert np.all(x_proj_over[25:] == 0)

    # 3. Exact cardinality: x already has 10 features, target is 10
    x_exact = np.zeros(n, dtype=np.int64)
    x_exact[:10] = 1
    x_proj_exact = project_cardinality_qubo(x_exact, r, d, K=10)
    assert np.array_equal(x_exact, x_proj_exact)


def test_qubo_selector_exact_cardinality_large_k():
    """Verify QUBOFeatureSelector guarantees exact cardinality K in [10, 20, 50]."""
    from src.selectors.qubo_selector import QUBOFeatureSelector

    np.random.seed(42)
    X = np.random.randn(30, 60)
    y = np.random.randint(0, 2, size=30)

    for K in [5, 15, 25]:
        selector = QUBOFeatureSelector(solver_name="QUBO-SA", K=K, seed=42)
        selector.fit(X, y)
        support = selector.get_support()
        assert np.sum(support) == K, f"QUBO-SA failed exact cardinality: {np.sum(support)} != {K}"
        assert selector.solver_info_["repaired_k"] == K
        assert "cardinality_satisfied_natively" in selector.solver_info_
        assert "raw_k" in selector.solver_info_

