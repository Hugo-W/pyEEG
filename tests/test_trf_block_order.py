"""
Focused tests for TRFEstimator block_order support (issue #28).

Verifies:
- constructor accepts/validates ``block_order`` ('lags' / 'features').
- ``_beta_to_coef`` / ``_coef_to_beta`` map solver columns to the canonical
  ``coef_`` shape (n_lags, n_feats, n_chans) for both orders.
- fit/predict produce identical results for both orders on the same raw data.
- pre-lagged inputs (lagged=True) work for feature-major layouts.
- the generated smoothness regularizer uses kron(L, I_feats) for lag-major and
  kron(I_feats, L) for feature-major, and zero-pads the intercept so the
  intercept stays unregularized.
- default ``block_order='lags'`` is backward compatible with the legacy layout.
"""

import numpy as np
import pytest

from pyeeg.models import TRFEstimator
from pyeeg.solvers import create_quadratic_regularizer
from pyeeg.utils import lag_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=300, n_feats=2, n_chans=2, n_lags=3, seed=0):
    """Small deterministic multi-feature/multi-channel dataset."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_feats))
    y = rng.standard_normal((n, n_chans))
    lags = np.arange(n_lags)  # 0, 1, 2 -> estimator lags are reversed internally
    return x, y, lags


def _fit(x, y, lags, block_order='lags', fit_intercept=False, alpha=None,
         quadratic_reg=None, lagged=False, X_lagged=None):
    """Fit a TRFEstimator on raw (or pre-lagged) data."""
    trf = TRFEstimator(times=lags.astype(float)[::-1], srate=1.0,
                       alpha=alpha, fit_intercept=fit_intercept,
                       quadratic_reg=quadratic_reg, block_order=block_order,
                       verbose=False)
    if lagged:
        trf.fit(X_lagged, y, lagged=True, drop=False)
    else:
        trf.fit(x, y, drop=False)
    return trf


def _manual_betas(trf, x, y):
    """Direct solve in the estimator's own block order (no intercept)."""
    X = lag_matrix(x, trf.lags, fill_value=0., block_order=trf.block_order)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------

class TestBlockOrderConstructor:
    def test_default_is_lags(self):
        trf = TRFEstimator(times=[0.], srate=1.0)
        assert trf.block_order == 'lags'

    def test_accepts_features(self):
        trf = TRFEstimator(times=[0.], srate=1.0, block_order='features')
        assert trf.block_order == 'features'

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="block_order"):
            TRFEstimator(times=[0.], srate=1.0, block_order='invalid')


# ---------------------------------------------------------------------------
# Beta <-> coef_ mapping helpers
# ---------------------------------------------------------------------------

class TestBetaCoefMapping:
    def _trf(self, block_order):
        trf = TRFEstimator(times=[2., 1., 0.], srate=1.0, block_order=block_order)
        trf.fill_lags()
        trf.n_feats_ = 2
        return trf

    def test_roundtrip_lags(self):
        trf = self._trf('lags')
        coef = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
        beta = trf._coef_to_beta(coef)
        # lag-major layout: interleaved per lag -> [f0_l0, f1_l0, f0_l1, f1_l1, ...]
        np.testing.assert_allclose(beta, coef[::-1].reshape(6, 4))
        np.testing.assert_allclose(trf._beta_to_coef(beta), coef)

    def test_roundtrip_features(self):
        trf = self._trf('features')
        coef = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
        beta = trf._coef_to_beta(coef)
        # feature-major layout: [f0_l0, f0_l1, f0_l2, f1_l0, f1_l1, f1_l2]
        expected = coef[::-1].swapaxes(0, 1).reshape(6, 4)
        np.testing.assert_allclose(beta, expected)
        np.testing.assert_allclose(trf._beta_to_coef(beta), coef)

    def test_orders_agree_on_same_coef(self):
        """The two layouts only permute columns: mapping the same coef_ back to
        either layout yields the same underlying coefficients."""
        trf_l = self._trf('lags')
        trf_f = self._trf('features')
        coef = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)
        np.testing.assert_allclose(trf_l._beta_to_coef(trf_l._coef_to_beta(coef)),
                                   trf_f._beta_to_coef(trf_f._coef_to_beta(coef)))


# ---------------------------------------------------------------------------
# Fit / predict equivalence across block orders
# ---------------------------------------------------------------------------

class TestBlockOrderFit:
    def test_fit_orders_identical_raw(self):
        """Fitting the same raw data with 'lags' and 'features' must yield the
        canonical coef_/intercept_ (the internal layout is handled internally)."""
        x, y, lags = _make_data()
        trf_l = _fit(x, y, lags, block_order='lags', fit_intercept=True)
        trf_f = _fit(x, y, lags, block_order='features', fit_intercept=True)
        np.testing.assert_allclose(trf_f.coef_, trf_l.coef_)
        np.testing.assert_allclose(trf_f.intercept_, trf_l.intercept_)

    def test_fit_matches_legacy_lags_solve(self):
        """Default (lags) fit must reproduce the legacy manual solve: lag the
        matrix in the lag-major layout, solve, reshape+flip."""
        x, y, lags = _make_data()
        trf = _fit(x, y, lags, block_order='lags')
        beta = _manual_betas(trf, x, y)
        np.testing.assert_allclose(trf.coef_, beta.reshape(3, 2, 2)[::-1],
                                   atol=1e-8)

    def test_fit_features_matches_manual_solve(self):
        """Feature-major fit must equal a manual solve in feature-major layout."""
        x, y, lags = _make_data()
        trf = _fit(x, y, lags, block_order='features')
        beta = _manual_betas(trf, x, y)
        np.testing.assert_allclose(trf.coef_, beta.reshape(2, 3, 2).swapaxes(0, 1)[::-1],
                                   atol=1e-8)

    def test_fit_lagged_features_matches_raw(self):
        """Pre-lagged feature-major input with lagged=True must match the raw fit."""
        x, y, lags = _make_data()
        trf_raw = _fit(x, y, lags, block_order='features')
        X_lag = lag_matrix(x, trf_raw.lags, fill_value=0., block_order='features')
        trf_lag = _fit(x, y, lags, block_order='features', lagged=True,
                       X_lagged=X_lag)
        np.testing.assert_allclose(trf_lag.coef_, trf_raw.coef_, atol=1e-8)

    def test_fit_lagged_lags_matches_raw(self):
        """Pre-lagged lag-major input with lagged=True must match the raw fit."""
        x, y, lags = _make_data()
        trf_raw = _fit(x, y, lags, block_order='lags')
        X_lag = lag_matrix(x, trf_raw.lags, fill_value=0., block_order='lags')
        trf_lag = _fit(x, y, lags, block_order='lags', lagged=True,
                       X_lagged=X_lag)
        np.testing.assert_allclose(trf_lag.coef_, trf_raw.coef_, atol=1e-8)

    def test_predict_orders_match(self):
        """predict() on raw X must agree across block orders and with predict on
        the pre-lagged matrices."""
        x, y, lags = _make_data(n=200)
        trf_l = _fit(x, y, lags, block_order='lags', fit_intercept=True)
        trf_f = _fit(x, y, lags, block_order='features', fit_intercept=True)
        yhat_l = trf_l.predict(x)
        yhat_f = trf_f.predict(x)
        np.testing.assert_allclose(yhat_f, yhat_l, atol=1e-8)

        # pre-lagged input (in the model's block order) must predict identically
        X_lag_l = lag_matrix(x, trf_l.lags, fill_value=0., block_order='lags')
        X_lag_f = lag_matrix(x, trf_f.lags, fill_value=0., block_order='features')
        # pass the intercept column in manually: predict() only re-lags when the
        # column count differs from the expected design size
        X_lag_l_i = np.hstack([np.ones((len(X_lag_l), 1)), X_lag_l])
        X_lag_f_i = np.hstack([np.ones((len(X_lag_f), 1)), X_lag_f])
        np.testing.assert_allclose(trf_l.predict(X_lag_l_i), yhat_l, atol=1e-8)
        np.testing.assert_allclose(trf_f.predict(X_lag_f_i), yhat_f, atol=1e-8)


# ---------------------------------------------------------------------------
# Quadratic regularizer ordering and intercept padding
# ---------------------------------------------------------------------------

class TestBlockOrderRegularizer:
    def test_regularizer_lags_order(self):
        """'lags' order must produce kron(L, I_feats)."""
        n_lags, n_feats = 3, 2
        trf = TRFEstimator(times=np.arange(n_lags)[::-1].astype(float), srate=1.0,
                           alpha=1.0, quadratic_reg='smoothness',
                           block_order='lags', fit_intercept=False)
        trf.fill_lags()
        trf.n_feats_ = n_feats
        L = create_quadratic_regularizer('smoothness', n_lags, alpha=1.0)
        expected = np.kron(L, np.eye(n_feats))
        np.testing.assert_allclose(trf._build_quadratic_regularizer(), expected)

    def test_regularizer_features_order(self):
        """'features' order must produce kron(I_feats, L)."""
        n_lags, n_feats = 3, 2
        trf = TRFEstimator(times=np.arange(n_lags)[::-1].astype(float), srate=1.0,
                           alpha=1.0, quadratic_reg='smoothness',
                           block_order='features', fit_intercept=False)
        trf.fill_lags()
        trf.n_feats_ = n_feats
        L = create_quadratic_regularizer('smoothness', n_lags, alpha=1.0)
        expected = np.kron(np.eye(n_feats), L)
        np.testing.assert_allclose(trf._build_quadratic_regularizer(), expected)

    def test_regularizer_intercept_zero_padded(self):
        """fit_intercept=True must zero-pad M so the intercept is unregularized."""
        n_lags, n_feats = 3, 2
        trf = TRFEstimator(times=np.arange(n_lags)[::-1].astype(float), srate=1.0,
                           alpha=1.0, quadratic_reg='smoothness',
                           block_order='lags', fit_intercept=True)
        trf.fill_lags()
        trf.n_feats_ = n_feats
        L = create_quadratic_regularizer('smoothness', n_lags, alpha=1.0)
        inner = np.kron(L, np.eye(n_feats))
        expected = np.pad(inner, ((1, 0), (1, 0)))
        M = trf._build_quadratic_regularizer()
        assert M.shape == (n_lags * n_feats + 1, n_lags * n_feats + 1)
        np.testing.assert_allclose(M, expected)
        # intercept row/column must be untouched by regularization
        np.testing.assert_allclose(M[0, :], 0.0)
        np.testing.assert_allclose(M[:, 0], 0.0)

    def test_intercept_quadratic_reg_matches_direct_solve(self):
        """Fit with quadratic_reg + fit_intercept=True equals a direct solve of
        (XtX + M_padded) beta = XtY, for both block orders (issue #28 req. 4)."""
        for block_order in ('lags', 'features'):
            x, y, lags = _make_data(n=150, seed=1)
            trf = TRFEstimator(times=lags.astype(float)[::-1], srate=1.0,
                               alpha=2.0, quadratic_reg='smoothness',
                               fit_intercept=True, block_order=block_order,
                               verbose=False)
            trf.fit(x, y, drop=False)

            X = lag_matrix(x, trf.lags, fill_value=0., block_order=block_order)
            Xw = np.hstack([np.ones((len(X), 1)), X])
            L = create_quadratic_regularizer('smoothness', len(trf.lags), alpha=2.0)
            if block_order == 'lags':
                inner = np.kron(L, np.eye(trf.n_feats_))
            else:
                inner = np.kron(np.eye(trf.n_feats_), L)
            M = np.pad(inner, ((1, 0), (1, 0)))
            beta = np.linalg.solve(Xw.T @ Xw + M, Xw.T @ y)

            np.testing.assert_allclose(trf.intercept_, beta[0], atol=1e-8)
            np.testing.assert_allclose(trf.coef_, trf._beta_to_coef(beta[1:]),
                                       atol=1e-8)

    def test_intercept_quadratic_reg_list_matches_direct_solve(self):
        """List-of-arrays fit with quadratic_reg + fit_intercept must also
        zero-pad M and weight the intercept consistently (issue #28 req. 4/5)."""
        block_order = 'features'
        rng = np.random.default_rng(7)
        n_a, n_b = 120, 80
        xa = rng.standard_normal((n_a, 2))
        xb = rng.standard_normal((n_b, 2))
        ya = rng.standard_normal((n_a, 1))
        yb = rng.standard_normal((n_b, 1))
        wa = rng.uniform(0.1, 2.0, n_a)
        wb = rng.uniform(0.1, 2.0, n_b)
        lags = np.arange(3)

        trf = TRFEstimator(times=lags.astype(float)[::-1], srate=1.0,
                           alpha=2.0, quadratic_reg='smoothness',
                           fit_intercept=True, block_order=block_order,
                           verbose=False)
        trf.fit([xa, xb], [ya, yb], drop=False, weights=[wa, wb])

        # reference: intercept added before sqrt-weighting, M zero-padded for it
        Xa = lag_matrix(xa, trf.lags, fill_value=0., block_order=block_order)
        Xb = lag_matrix(xb, trf.lags, fill_value=0., block_order=block_order)
        swa, swb = np.sqrt(wa), np.sqrt(wb)
        Xw = np.vstack([np.hstack([np.ones((n_a, 1)), Xa]) * swa[:, None],
                        np.hstack([np.ones((n_b, 1)), Xb]) * swb[:, None]])
        yw = np.concatenate([ya * swa[:, None], yb * swb[:, None]])
        L = create_quadratic_regularizer('smoothness', len(trf.lags), alpha=2.0)
        inner = np.kron(np.eye(trf.n_feats_), L)
        M = np.pad(inner, ((1, 0), (1, 0)))
        beta = np.linalg.solve(Xw.T @ Xw + M, Xw.T @ yw)

        np.testing.assert_allclose(trf.intercept_, beta[0], atol=1e-8)
        np.testing.assert_allclose(trf.coef_, trf._beta_to_coef(beta[1:]),
                                   atol=1e-8)