"""Tests for per-feature (banded) ridge regularization (issue #18)."""

import numpy as np
import pytest

from pyeeg.models import TRFEstimator
from pyeeg.utils import lag_matrix


def test_feature_alphas_build_diagonal_for_both_block_orders():
    for block_order, expected in (
        ('lags', [2., 10., 2., 10., 2., 10.]),
        ('features', [2., 2., 2., 10., 10., 10.]),
    ):
        trf = TRFEstimator(times=[2., 1., 0.], feature_alphas=[2., 10.],
                           fit_intercept=False, block_order=block_order)
        trf.fill_lags()
        trf.n_feats_ = 2
        np.testing.assert_allclose(
            np.diag(trf._build_quadratic_regularizer()), expected)


def test_feature_alphas_matches_direct_banded_ridge_solve():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((200, 2))
    y = rng.standard_normal((200, 1))
    lags = np.array([2., 1., 0.])

    for block_order in ('lags', 'features'):
        trf = TRFEstimator(times=lags, feature_alphas=[0.5, 20.],
                           fit_intercept=False, block_order=block_order,
                           verbose=False)
        trf.fit(x, y, drop=False)
        design = lag_matrix(x, lags=trf.lags, mode='full', fill_value=0.,
                            block_order=block_order)
        diagonal = np.diag(trf._build_quadratic_regularizer())
        expected = np.linalg.solve(
            design.T @ design + np.diag(diagonal), design.T @ y)
        np.testing.assert_allclose(trf.coef_, trf._beta_to_coef(expected),
                                   atol=1e-10)


def test_feature_alphas_requires_one_value_per_feature():
    trf = TRFEstimator(times=[1., 0.], feature_alphas=[1., 2., 3.],
                       fit_intercept=False, verbose=False)
    with pytest.raises(ValueError, match="one value per input feature"):
        trf.fit(np.ones((10, 2)), np.ones((10, 1)), drop=False)


def test_feature_alphas_are_incompatible_with_quadratic_reg_and_alpha_path():
    with pytest.raises(ValueError, match="quadratic_reg"):
        TRFEstimator(times=[0.], feature_alphas=[1.], quadratic_reg='smoothness')
    with pytest.raises(ValueError, match="alpha path"):
        TRFEstimator(times=[0.], feature_alphas=[1.], alpha=[1., 2.])


def test_feature_alphas_work_with_segmented_fit():
    rng = np.random.default_rng(3)
    x = [rng.standard_normal((40, 2)), rng.standard_normal((30, 2))]
    y = [rng.standard_normal((40, 1)), rng.standard_normal((30, 1))]
    trf = TRFEstimator(times=[1., 0.], feature_alphas=[0.5, 4.],
                       fit_intercept=True, block_order='features', verbose=False)
    trf.fit(x, y, drop=False)
    assert trf.coef_.shape == (2, 2, 1)
    np.testing.assert_allclose(np.diag(trf._build_quadratic_regularizer()),
                               [0., 0.5, 0.5, 4., 4.])


def test_feature_alphas_work_with_robust_irls():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((80, 2))
    y = rng.standard_normal((80, 1))
    trf = TRFEstimator(times=[0.], feature_alphas=[0.5, 2.], fit_intercept=False,
                       loss='cauchy', robust_solver='irls', verbose=False)
    trf.fit(x, y, drop=False)
    assert trf.coef_.shape == (1, 2, 1)


def test_feature_alphas_reject_nonlinear_robust_solver():
    trf = TRFEstimator(times=[0.], feature_alphas=[1.], fit_intercept=False,
                       loss='cauchy', robust_solver='least_squares',
                       verbose=False)
    with pytest.raises(ValueError, match="least_squares"):
        trf.fit(np.ones((10, 1)), np.ones((10, 1)), drop=False)
