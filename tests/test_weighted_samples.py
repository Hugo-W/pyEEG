"""
Tests for weighted-sample TRF estimation (issue #17).

Verifies:
- ``apply_sample_weights`` row-scales X and y by sqrt(weights).
- ``fit(weights=...)`` solves weighted least squares on the single-array path.
- ``fit(weights=...)`` solves weighted least squares on the list-of-arrays path.
- weights=ones reproduces the unweighted fit (backward compatibility).
- negative weights are rejected.
- weighted fit composes with quadratic regularization.
"""

import numpy as np
import pytest

from pyeeg.utils import apply_sample_weights

# ---------------------------------------------------------------------------
# apply_sample_weights unit tests
# ---------------------------------------------------------------------------


class TestApplySampleWeights:
    def test_ones_weights_identity(self):
        """sqrt(1) scaling: weights=ones leaves X and y unchanged."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal((20, 2))
        Xw, yw = apply_sample_weights(X, y, np.ones(20))
        np.testing.assert_allclose(Xw, X)
        np.testing.assert_allclose(yw, y)

    def test_row_scaling(self):
        """Each row i is scaled by sqrt(w[i])."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10,))
        w = rng.uniform(0.5, 2.0, 10)
        Xw, yw = apply_sample_weights(X, y, w)
        sw = np.sqrt(w)
        np.testing.assert_allclose(Xw, X * sw[:, None])
        np.testing.assert_allclose(yw, y * sw)

    def test_1d_y(self):
        """1-D y is supported and returned as 1-D."""
        X = np.ones((5, 2))
        y = np.arange(5.0)
        w = np.full(5, 4.0)  # sqrt(4) = 2
        _, yw = apply_sample_weights(X, y, w)
        np.testing.assert_allclose(yw, y * 2.0)


# ---------------------------------------------------------------------------
# TRFEstimator weighted fit tests
# ---------------------------------------------------------------------------


def _wls_reference(X, y, w, intercept=True):
    """Direct weighted least squares solve, returning (intercept, coefs).

    The intercept column is added to X *before* sqrt-W row scaling, so it is
    weighted like every other column (correct WLS semantics).
    """
    if intercept:
        X = np.hstack([np.ones((len(X), 1)), X])
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    if intercept:
        return beta[0], beta[1:]
    return None, beta


class TestTRFWeightedFit:
    def test_ones_weights_match_unweighted(self):
        """weights=ones must reproduce the unweighted fit exactly."""
        from pyeeg.models import TRFEstimator
        from pyeeg.simulate import (
            dummy_trf_kernel,
            simulate_pulse_inputs,
            simulate_trf_output,
        )
        from pyeeg.utils import lag_matrix

        srate = 100
        tker, ker = dummy_trf_kernel(srate=srate)
        _, x = simulate_pulse_inputs(n_events=200, dur=30.0, srate=srate, seed=42)
        y = simulate_trf_output(tker, ker, x, srate=srate)
        lags = np.round(tker * srate).astype(int)
        X = lag_matrix(x[:, None], lags, mode="full", fill_value=0.0)

        trf0 = TRFEstimator(
            times=tker, srate=srate, alpha=1.0, fit_intercept=False, verbose=False
        )
        trf0.fit(X[:, ::-1], y[:, None], lagged=True, drop=False)

        trf1 = TRFEstimator(
            times=tker, srate=srate, alpha=1.0, fit_intercept=False, verbose=False
        )
        trf1.fit(
            X[:, ::-1], y[:, None], lagged=True, drop=False, weights=np.ones(len(y))
        )

        np.testing.assert_allclose(trf0.coef_, trf1.coef_)

    def test_weighted_matches_direct_wls(self):
        """Single-array weighted fit must match a direct WLS solve."""
        from pyeeg.models import TRFEstimator

        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(n)
        w = rng.uniform(0.1, 2.0, n)

        ref_intercept, ref_coef = _wls_reference(X, y, w, intercept=True)

        trf = TRFEstimator(
            times=[0.0], srate=1.0, alpha=None, fit_intercept=True, verbose=False
        )
        trf.fit(X, y[:, None], drop=False, weights=w)

        np.testing.assert_allclose(trf.coef_.squeeze(), ref_coef, atol=1e-8)
        np.testing.assert_allclose(trf.intercept_, ref_intercept, atol=1e-8)

    def test_weighted_no_intercept_matches_direct_wls(self):
        """Weighted fit without intercept matches direct WLS (no intercept)."""
        from pyeeg.models import TRFEstimator

        rng = np.random.default_rng(2)
        n = 150
        X = rng.standard_normal((n, 4))
        y = X @ np.array([0.5, -1.0, 2.0, 0.25]) + 0.2 * rng.standard_normal(n)
        w = rng.uniform(0.05, 3.0, n)

        _, ref_coef = _wls_reference(X, y, w, intercept=False)

        trf = TRFEstimator(
            times=[0.0], srate=1.0, alpha=None, fit_intercept=False, verbose=False
        )
        trf.fit(X, y[:, None], drop=False, weights=w)

        np.testing.assert_allclose(trf.coef_.squeeze(), ref_coef, atol=1e-8)

    def test_weighted_list_matches_direct_wls(self):
        """List-of-arrays weighted fit must match a stacked direct WLS solve."""
        from pyeeg.models import TRFEstimator

        rng = np.random.default_rng(3)
        Xa = rng.standard_normal((120, 2))
        Xb = rng.standard_normal((80, 2))
        ya = Xa @ np.array([1.0, -1.0]) + 0.1 * rng.standard_normal(120)
        yb = Xb @ np.array([1.0, -1.0]) + 0.1 * rng.standard_normal(80)
        wa = rng.uniform(0.1, 2.0, 120)
        wb = rng.uniform(0.1, 2.0, 80)

        # reference: stack weighted segments and solve once.
        # The intercept column is added to X *before* sqrt-W row scaling, so it
        # is weighted like every other column (correct WLS semantics, consistent
        # with the single-array path and _wls_reference).
        swa, swb = np.sqrt(wa), np.sqrt(wb)
        Xref = np.vstack(
            [
                np.hstack([np.ones((120, 1)), Xa]) * swa[:, None],
                np.hstack([np.ones((80, 1)), Xb]) * swb[:, None],
            ]
        )
        yref = np.concatenate([ya * swa, yb * swb])
        beta = np.linalg.lstsq(Xref, yref, rcond=None)[0]
        ref_intercept, ref_coef = beta[0], beta[1:]

        trf = TRFEstimator(
            times=[0.0], srate=1.0, alpha=None, fit_intercept=True, verbose=False
        )
        trf.fit([Xa, Xb], [ya[:, None], yb[:, None]], drop=False, weights=[wa, wb])

        np.testing.assert_allclose(trf.coef_.squeeze(), ref_coef, atol=1e-8)
        np.testing.assert_allclose(trf.intercept_, ref_intercept, atol=1e-8)

    def test_weighted_list_lagged_matches_unweighted_ones(self):
        """List path with lagged=True: weights=ones must match the unweighted
        fit (backward compatibility for the lagged list path)."""
        from pyeeg.models import TRFEstimator
        from pyeeg.utils import lag_matrix

        rng = np.random.default_rng(4)
        srate = 100
        lags = np.array([0, 1, 2])
        n_a, n_b = 200, 150
        xa = rng.standard_normal((n_a, 1))
        xb = rng.standard_normal((n_b, 1))
        ya = rng.standard_normal((n_a, 1))
        yb = rng.standard_normal((n_b, 1))

        Xa = lag_matrix(xa, lags, mode="full", fill_value=0.0)
        Xb = lag_matrix(xb, lags, mode="full", fill_value=0.0)

        trf0 = TRFEstimator(
            times=lags.astype(float)[::-1] / srate,
            srate=srate,
            alpha=1.0,
            fit_intercept=True,
            verbose=False,
        )
        trf0.fit([Xa[:, ::-1], Xb[:, ::-1]], [ya, yb], lagged=True, drop=False)

        trf1 = TRFEstimator(
            times=lags.astype(float)[::-1] / srate,
            srate=srate,
            alpha=1.0,
            fit_intercept=True,
            verbose=False,
        )
        trf1.fit(
            [Xa[:, ::-1], Xb[:, ::-1]],
            [ya, yb],
            lagged=True,
            drop=False,
            weights=[np.ones(n_a), np.ones(n_b)],
        )

        np.testing.assert_allclose(trf0.coef_, trf1.coef_)
        np.testing.assert_allclose(trf0.intercept_, trf1.intercept_)

    def test_negative_weights_rejected(self):
        """Negative weights must raise ValueError."""
        from pyeeg.models import TRFEstimator

        rng = np.random.default_rng(5)
        n = 50
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal((n, 1))
        w = np.ones(n)
        w[5] = -0.1

        trf = TRFEstimator(
            times=[0.0], srate=1.0, alpha=None, fit_intercept=False, verbose=False
        )
        with pytest.raises(ValueError, match="non-negative"):
            trf.fit(X, y, drop=False, weights=w)

    def test_weighted_with_quadratic_reg(self):
        """Weighted fit must compose with quadratic regularization.

        Uses a single feature (consistent with the existing quadratic-reg
        tests) because the multi-feature coefficient layout depends on the
        lag-matrix block order, which is orthogonal to this issue.
        """
        from pyeeg.models import TRFEstimator
        from pyeeg.solvers import create_quadratic_regularizer
        from pyeeg.utils import apply_sample_weights, lag_matrix

        rng = np.random.default_rng(6)
        n = 300
        srate = 100
        x = rng.standard_normal((n, 1))
        y = rng.standard_normal((n, 1))
        w = rng.uniform(0.1, 2.0, n)

        trf = TRFEstimator(
            tmin=-0.02,
            tmax=0.02,
            srate=srate,
            alpha=10.0,
            quadratic_reg="smoothness",
            fit_intercept=False,
            verbose=False,
        )
        trf.fill_lags()
        trf.fit(x, y, drop=False, weights=w)

        # reference: weighted X/y then solve (XtX + alpha*M) beta = XtY.
        X = lag_matrix(x, trf.lags, mode="full", fill_value=0.0)
        Xw, yw = apply_sample_weights(X, y, w)
        M = create_quadratic_regularizer("smoothness", len(trf.lags), alpha=10.0)
        ref = np.linalg.solve(Xw.T @ Xw + M, Xw.T @ yw)
        # coef_ is stored lag-flipped relative to the solve order
        np.testing.assert_allclose(trf.coef_[:, 0, 0], ref[::-1, 0], atol=1e-6)
