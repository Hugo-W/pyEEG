"""Tests for the Solver class hierarchy (pyeeg.solvers) and the ``solver``
parameter of :class:`pyeeg.models.trf.TRFEstimator`.
"""

import numpy as np
import pytest

from pyeeg.solvers import (
    Solver,
    SolverResult,
    SVDSolver,
    LSTSQSolver,
    ConjugateGradientSolver,
    IRLSSolver,
    ScipyRobustSolver,
    _svd_regress,
    _lstsq_regress,
    _robust_irls_regress,
    _robust_least_squares_regress,
)


# ---------------------------------------------------------------------------
# Solver ABC and SolverResult
# ---------------------------------------------------------------------------
def test_solver_abc_cannot_instantiate():
    """Solver is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Solver()


def test_solver_result_dataclass():
    """SolverResult holds betas and optional info."""
    betas = np.random.randn(5, 2)
    result = SolverResult(betas)
    assert result.betas is betas
    assert result.info is None
    info = {"n_iter": 10}
    result2 = SolverResult(betas, info)
    assert result2.info is info


# ---------------------------------------------------------------------------
# SVDSolver
# ---------------------------------------------------------------------------
def test_svd_solver_basic():
    """SVDSolver.solve() returns SolverResult with correct betas."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_betas = np.random.randn(5, 2)
    y = X @ true_betas + 0.01 * np.random.randn(100, 2)
    solver = SVDSolver()
    result = solver.solve(X, y, alpha=0.1)
    assert isinstance(result, SolverResult)
    assert result.info is None
    assert result.betas.shape == (5, 2, 1)  # (n_feats, n_chans, n_alphas)
    # Recovered betas should be close to true betas
    np.testing.assert_allclose(result.betas[..., 0], true_betas, atol=0.05)


def test_svd_solver_matches_free_function():
    """SVDSolver.solve().betas matches _svd_regress() output."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randn(50, 3)
    alpha = [0.1, 1.0, 10.0]
    # Free function
    betas_free = _svd_regress(X, y, alpha)
    # Class
    solver = SVDSolver()
    betas_class = solver.solve(X, y, alpha).betas
    np.testing.assert_allclose(betas_class, betas_free)


def test_svd_solver_with_M():
    """SVDSolver handles quadratic regularization matrix M."""
    np.random.seed(42)
    X = np.random.randn(80, 6)
    y = np.random.randn(80, 2)
    M = np.diag([1.0] * 6)
    solver = SVDSolver()
    result = solver.solve(X, y, alpha=1.0, M=M)
    assert result.betas.shape == (6, 2, 1)
    # Should match free function
    betas_free = _svd_regress(X, y, 1.0, M=M)
    np.testing.assert_allclose(result.betas, betas_free)


def test_svd_solver_segmented():
    """SVDSolver handles list-of-arrays (segmented) inputs."""
    np.random.seed(42)
    X1 = np.random.randn(60, 3)
    X2 = np.random.randn(40, 3)
    y1 = np.random.randn(60, 2)
    y2 = np.random.randn(40, 2)
    solver = SVDSolver()
    result = solver.solve([X1, X2], [y1, y2], alpha=1.0)
    betas_free = _svd_regress([X1, X2], [y1, y2], 1.0)
    np.testing.assert_allclose(result.betas, betas_free)


# ---------------------------------------------------------------------------
# LSTSQSolver
# ---------------------------------------------------------------------------
def test_lstsq_solver_basic():
    """LSTSQSolver.solve() returns SolverResult with correct betas."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_betas = np.random.randn(5, 2)
    y = X @ true_betas + 0.01 * np.random.randn(100, 2)
    solver = LSTSQSolver()
    result = solver.solve(X, y)
    assert isinstance(result, SolverResult)
    assert result.info is None
    assert result.betas.shape == (5, 2)
    np.testing.assert_allclose(result.betas, true_betas, atol=0.05)


def test_lstsq_solver_matches_free_function():
    """LSTSQSolver.solve().betas matches _lstsq_regress() output."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.randn(50, 3)
    betas_free = _lstsq_regress(X, y)
    betas_class = LSTSQSolver().solve(X, y).betas
    np.testing.assert_allclose(betas_class, betas_free)


# ---------------------------------------------------------------------------
# ConjugateGradientSolver
# ---------------------------------------------------------------------------
def test_cg_solver_matches_lstsq():
    """ConjugateGradientSolver with no regularization matches OLS."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randn(50, 2)
    cg_solver = ConjugateGradientSolver(tol=1e-12)
    result = cg_solver.solve(X, y, alpha=0.0)
    ols_betas = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(result.betas, ols_betas, atol=1e-6)


# ---------------------------------------------------------------------------
# IRLSSolver
# ---------------------------------------------------------------------------
def test_irls_solver_basic():
    """IRLSSolver.solve() returns SolverResult with info dict."""
    np.random.seed(42)
    X = np.random.randn(200, 4)
    true_betas = np.random.randn(4, 2)
    y = X @ true_betas + 0.1 * np.random.randn(200, 2)
    y[::25] += 12.0  # add outliers
    solver = IRLSSolver(max_iter=40, tol=1e-8)
    result = solver.solve(X, y, alpha=0.0)
    assert isinstance(result, SolverResult)
    assert result.info is not None
    assert "n_iter" in result.info
    assert "converged" in result.info
    assert "scale" in result.info
    assert result.betas.shape == (4, 2)
    # Robust betas should be closer to true betas than OLS
    ols_betas = np.linalg.lstsq(X, y, rcond=None)[0]
    robust_error = np.mean((result.betas - true_betas) ** 2)
    ols_error = np.mean((ols_betas - true_betas) ** 2)
    assert robust_error < ols_error


def test_irls_solver_matches_free_function():
    """IRLSSolver.solve() matches _robust_irls_regress() output."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100, 2)
    # Free function
    betas_free, info_free = _robust_irls_regress(X, y, alpha=0.1, max_iter=10)
    # Class
    solver = IRLSSolver(max_iter=10)
    result = solver.solve(X, y, alpha=0.1)
    np.testing.assert_allclose(result.betas, betas_free)
    assert result.info["n_iter"] == info_free["n_iter"]
    assert result.info["converged"] == info_free["converged"]


# ---------------------------------------------------------------------------
# ScipyRobustSolver
# ---------------------------------------------------------------------------
def test_scipy_robust_solver_basic():
    """ScipyRobustSolver.solve() returns SolverResult with info dict."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100, 2)
    solver = ScipyRobustSolver()
    result = solver.solve(X, y)
    assert isinstance(result, SolverResult)
    assert result.info is not None
    assert result.betas.shape == (3, 2)


def test_scipy_robust_solver_matches_free_function():
    """ScipyRobustSolver.solve() matches _robust_least_squares_regress() output."""
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = np.random.randn(80, 2)
    betas_free, info_free = _robust_least_squares_regress(X, y)
    solver = ScipyRobustSolver()
    result = solver.solve(X, y)
    np.testing.assert_allclose(result.betas, betas_free, atol=1e-10)


# ---------------------------------------------------------------------------
# TRFEstimator with solver parameter
# ---------------------------------------------------------------------------
def _make_trf_data(srate=100, n_events=200, seed=42):
    """Generate a pulse-input TRF dataset: X (n, 1), Y (n, 1)."""
    from pyeeg.simulate import (
        dummy_trf_kernel,
        simulate_pulse_inputs,
        simulate_trf_output,
    )

    tker, ker = dummy_trf_kernel(srate=srate)
    _, x = simulate_pulse_inputs(n_events=n_events, dur=30.0, srate=srate, seed=seed)
    y = simulate_trf_output(tker, ker, x, srate=srate)
    rng = np.random.default_rng(seed)
    y = y + 0.1 * rng.standard_normal(len(y))
    return x[:, None], y[:, None], tker, ker


def test_trf_with_svd_solver_matches_default():
    """TRFEstimator with solver=SVDSolver() produces same result as default."""
    from pyeeg.models.trf import TRFEstimator

    np.random.seed(42)
    srate = 100
    X, Y, _, _ = _make_trf_data(srate=srate)
    # Default (implicit dispatch)
    trf_default = TRFEstimator(
        tmin=-0.1, tmax=0.3, srate=srate, alpha=1.0, verbose=False
    )
    trf_default.fit(X, Y, lagged=False, drop=False)
    # Explicit solver
    trf_solver = TRFEstimator(
        tmin=-0.1, tmax=0.3, srate=srate, alpha=1.0, verbose=False, solver=SVDSolver()
    )
    trf_solver.fit(X, Y, lagged=False, drop=False)
    np.testing.assert_allclose(trf_solver.coef_, trf_default.coef_, atol=1e-12)


def test_trf_with_lstsq_solver():
    """TRFEstimator with solver=LSTSQSolver() produces OLS result."""
    from pyeeg.models.trf import TRFEstimator

    np.random.seed(42)
    srate = 100
    X, Y, _, _ = _make_trf_data(srate=srate)
    trf = TRFEstimator(
        tmin=-0.1, tmax=0.3, srate=srate, alpha=0.0, verbose=False, solver=LSTSQSolver()
    )
    trf.fit(X, Y, lagged=False, drop=False)
    assert trf.coef_.shape == (len(trf.lags), 1, 1)
    assert trf.fitted


def test_trf_with_irls_solver():
    """TRFEstimator with solver=IRLSSolver() produces robust fit with metadata."""
    from pyeeg.models.trf import TRFEstimator

    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = np.random.randn(300, 2)
    trf = TRFEstimator(
        tmin=0.0, tmax=0.01, srate=100, verbose=False, solver=IRLSSolver(max_iter=20)
    )
    trf.fit(X, y, lagged=False, drop=False)
    assert trf.fitted
    assert trf.robust_n_iter_ is not None
    assert trf.robust_converged_ is not None
    assert trf.robust_scale_ is not None


def test_trf_rejects_non_solver():
    """TRFEstimator rejects non-Solver objects."""
    from pyeeg.models.trf import TRFEstimator

    with pytest.raises(ValueError, match="solver must be a Solver instance"):
        TRFEstimator(solver="not_a_solver")
    with pytest.raises(ValueError, match="solver must be a Solver instance"):
        TRFEstimator(solver=42)


def test_trf_solver_none_is_default():
    """TRFEstimator with solver=None uses the default implicit dispatch."""
    from pyeeg.models.trf import TRFEstimator

    trf = TRFEstimator(tmin=-0.1, tmax=0.1, srate=100, alpha=1.0, verbose=False)
    assert trf.solver is None