"""Tests for robust Cauchy-loss TRF estimation."""

import numpy as np
import pytest

from pyeeg.models import TRFEstimator
from pyeeg.solvers import (
    _robust_irls_regress,
    _robust_least_squares_regress,
)


def _outlier_data(seed=0, n_samples=400, n_features=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    beta = np.array([1.0, -1.5, 0.75, 0.25])[:n_features]
    y = X @ beta + 0.1 * rng.standard_normal(n_samples)
    y[::25] += 12.0
    return X, y[:, None], beta


def test_irls_downweights_gross_outliers():
    X, y, beta = _outlier_data()
    estimate, info = _robust_irls_regress(
        X, y, scale=0.2, max_iter=40, tol=1e-8)

    ordinary = np.linalg.lstsq(X, y, rcond=None)[0][:, 0]
    assert np.linalg.norm(estimate[:, 0] - beta) < np.linalg.norm(ordinary - beta)
    assert info['converged']
    assert info['n_iter'] < 40


def test_scipy_cauchy_path_agrees_with_irls():
    X, y, _ = _outlier_data(seed=1)
    irls, _ = _robust_irls_regress(X, y, scale=0.2, max_iter=50)
    scipy_fit, info = _robust_least_squares_regress(X, y, scale=0.2)

    np.testing.assert_allclose(irls, scipy_fit, atol=2e-4)
    assert info['converged']


def test_estimator_preserves_multichannel_shape_and_skips_ols_stats():
    X, y, _ = _outlier_data(seed=2)
    y_multi = np.hstack([y, 0.5 * y + 0.1 * np.random.default_rng(3).standard_normal(y.shape)])
    trf = TRFEstimator(
        times=[0.], fit_intercept=True, loss='cauchy', robust_sigma=0.2,
        robust_max_iter=40, verbose=False)
    trf.fit(X, y_multi, drop=False)

    assert trf.coef_.shape == (1, X.shape[1], 2)
    assert trf.intercept_.shape == (2,)
    assert trf.tvals_ is None
    assert trf.pvals_ is None
    assert trf.robust_converged_
    assert trf.robust_scale_.shape == (2,)


def test_estimator_irls_list_matches_single_array():
    X, y, _ = _outlier_data(seed=4)
    split = 215
    kwargs = dict(times=[0.], fit_intercept=False, loss='cauchy',
                  robust_sigma=0.2, verbose=False)
    single = TRFEstimator(**kwargs)
    single.fit(X, y, drop=False)

    segmented = TRFEstimator(**kwargs)
    segmented.fit([X[:split], X[split:]], [y[:split], y[split:]], drop=False)

    np.testing.assert_allclose(segmented.coef_, single.coef_, atol=1e-8)


def test_estimator_supports_quadratic_regularization_and_cg_inner_solver():
    X, y, _ = _outlier_data(seed=5)
    trf = TRFEstimator(
        times=[0.], alpha=1.0, fit_intercept=False, loss='cauchy',
        robust_sigma=0.2, robust_inner_solver='cg', verbose=False)
    trf.fit(X, y, drop=False)

    assert trf.coef_.shape == (1, X.shape[1], 1)
    assert trf.robust_converged_


def test_least_squares_path_rejects_regularization_and_sample_weights():
    X, y, _ = _outlier_data(seed=6)
    with pytest.raises(ValueError, match='unregularized'):
        TRFEstimator(times=[0.], alpha=1., loss='cauchy',
                     robust_solver='least_squares', verbose=False).fit(
                         X, y, drop=False)

    with pytest.raises(ValueError, match='weights'):
        TRFEstimator(times=[0.], loss='cauchy', verbose=False).fit(
            X, y, drop=False, weights=np.ones(len(y)))


def test_linear_loss_remains_backward_compatible():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 2))
    y = (X @ np.array([1.0, -2.0]) + 0.1 * rng.standard_normal(100))[:, None]
    old = TRFEstimator(times=[0.], fit_intercept=False, verbose=False)
    new = TRFEstimator(times=[0.], fit_intercept=False, loss='linear', verbose=False)
    old.fit(X, y, drop=False)
    new.fit(X, y, drop=False)
    np.testing.assert_allclose(old.coef_, new.coef_)
