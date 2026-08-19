"""
Simulation-based regression tests for TRF and CCA.

Uses known ground-truth kernels/relationships to verify that
TRFEstimator and CCA_Estimator can recover them from simulated data.
"""

import numpy as np
import pytest
from pyeeg.simulate import (
    dummy_trf_kernel,
    simulate_pulse_inputs,
    simulate_smooth_input,
    simulate_trf_output,
    simulate_ar,
    simulate_var,
)
from pyeeg.utils import lag_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trf_data(srate=100, dur=30.0, n_events=200, seed=42,
                   kernel_type='gaussian', tmin=-0.2, tmax=0.5, tloc=0.1, sigma=0.1):
    """Build a full TRF dataset with known kernel."""
    tker, ker = dummy_trf_kernel(tmin=tmin, tmax=tmax, srate=srate,
                                 tloc=tloc, sigma=sigma, kernel_type=kernel_type)
    t, x = simulate_pulse_inputs(n_events=n_events, dur=dur, srate=srate, seed=seed)
    y = simulate_trf_output(tker, ker, x, srate=srate)
    return t, x, y, tker, ker


def _fit_trf_lagged(tker, ker, x, y, srate=100, alpha=1.0, quadratic_reg=None):
    """Fit TRFEstimator using the pre-lagged code path.

    Uses alpha > 0 by default so regularization is on and the stats
    computation (which is not needed for kernel recovery) is skipped.
    """
    from pyeeg.models import TRFEstimator
    lags = np.round(tker * srate).astype(int)
    X = lag_matrix(x if x.ndim == 2 else x[:, None], lags, filling=0., drop_missing=False)
    trf = TRFEstimator(times=tker, srate=srate, alpha=alpha, fit_intercept=False,
                       quadratic_reg=quadratic_reg)
    trf.fit(X[:, ::-1], y if y.ndim == 2 else y[:, None], lagged=True, drop=False)
    return trf.coef_.squeeze()


def _make_cca_data(n=5000, n_feats=3, n_chans=4, noise=0.5, seed=42):
    """Build data with known linear relationships for CCA recovery."""
    rng = np.random.default_rng(seed)
    n_latent = min(n_feats, n_chans)
    Z = rng.standard_normal((n, n_latent))

    A = rng.standard_normal((n_latent, n_feats))
    x = Z @ A + noise * rng.standard_normal((n, n_feats))

    B = rng.standard_normal((n_latent, n_chans))
    y = Z @ B + noise * rng.standard_normal((n, n_chans))

    return x, y, Z, A, B


# ===========================================================================
# TRF tests
# ===========================================================================

class TestTRFEstimator:
    """Regression tests for TRFEstimator using simulated data."""

    def test_trf_recovers_gaussian_kernel_unregularised(self):
        """TRFEstimator with small alpha should closely recover a known kernel."""
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)
        coef = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=0.001)

        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.95, f"Kernel correlation {corr:.3f} < 0.95"

    def test_trf_recovers_gaussian_kernel_svd(self):
        """TRFEstimator with regularised SVD should still recover the kernel shape."""
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)
        coef = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=1.0)

        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.90, f"Regularised kernel correlation {corr:.3f} < 0.90"

    def test_trf_quadratic_regularization_smoothness(self):
        """Single-matrix fit with quadratic_reg='smoothness' must run and recover
        the kernel (issue #24: the M path crashed with a matmul dimension error).
        alpha is left None so it defaults to the M-strength knob (scale 1)."""
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)
        coef = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=None,
                               quadratic_reg='smoothness')

        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.95, f"Smoothness-regularised kernel correlation {corr:.3f} < 0.95"

    def test_trf_quadratic_regularization_scaled(self):
        """alpha scales M (the M-strength knob). The result must equal a direct
        solve of (X.T@X + alpha*M) beta = X.T@y."""
        from pyeeg.models import TRFEstimator
        from pyeeg.solvers import create_quadratic_regularizer
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)
        Y = y[:, None]

        trf = TRFEstimator(times=tker, srate=srate, fit_intercept=False,
                           alpha=100.0, quadratic_reg='smoothness')
        trf.fill_lags()
        X = lag_matrix(x[:, None], trf.lags, filling=0., drop_missing=False)
        trf.fit(X, Y, lagged=True, drop=False)
        coef = trf.coef_.squeeze()

        # reference: direct solve of (XtX + alpha*M) beta = XtY in the
        # estimator's own lag ordering
        M = np.kron(np.eye(1), create_quadratic_regularizer('smoothness',
                                                            len(trf.lags),
                                                            alpha=100.0))
        ref = np.linalg.solve(X.T @ X + M, X.T @ Y).squeeze()
        # coef_ is stored lag-flipped relative to the solve
        assert np.allclose(coef, ref[::-1]), "alpha-scaled M does not match direct solve"

    def test_trf_quadratic_regularization_zero_alpha_is_ols(self):
        """alpha=0 with quadratic_reg set must reduce to plain least squares
        (M scaled to zero, regularization path taken, no stats computed)."""
        from pyeeg.models import TRFEstimator
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)
        Y = y[:, None]

        trf = TRFEstimator(times=tker, srate=srate, fit_intercept=False,
                           alpha=0.0, quadratic_reg='smoothness')
        trf.fill_lags()
        X = lag_matrix(x[:, None], trf.lags, filling=0., drop_missing=False)
        trf.fit(X, Y, lagged=True, drop=False)
        coef = trf.coef_.squeeze()

        # reference: plain OLS via direct lstsq in the estimator's lag ordering
        ref = np.linalg.lstsq(X, Y, rcond=None)[0].squeeze()
        assert np.allclose(coef, ref[::-1]), "alpha=0 with M should equal plain OLS"

    def _fit_trf_stats(self, fit_intercept, n_chans=1, seed=42):
        """Fit an unregularized TRF (alpha=None -> OLS) and return coef_/tvals_/pvals_."""
        from pyeeg.models import TRFEstimator
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=seed)
        Y = y[:, None] if n_chans == 1 else np.concatenate([y[:, None]] * n_chans, axis=1)
        trf = TRFEstimator(times=tker, srate=srate, fit_intercept=fit_intercept,
                           alpha=None, verbose=False)
        trf.fill_lags()
        X = lag_matrix(x[:, None], trf.lags, filling=0., drop_missing=False)
        trf.fit(X, Y, lagged=True, drop=False)
        return trf, X, Y

    def _coef_flat(self, trf):
        """coef_ flattened over lags x features, keeping the channel axis."""
        return trf.coef_.reshape(-1, trf.coef_.shape[-1])

    def test_trf_stats_no_intercept(self):
        """Unregularized fit with fit_intercept=False must compute t-/p-values
        (issue #25: the unconditional [1:, :] strip crashed)."""
        trf, X, Y = self._fit_trf_stats(fit_intercept=False)
        # tvals_/pvals_ must match the flattened coef_ (n_coefs, n_chans)
        assert trf.tvals_.shape == self._coef_flat(trf).shape, \
            f"tvals_ {trf.tvals_.shape} != coef_ {self._coef_flat(trf).shape}"
        assert trf.pvals_.shape == self._coef_flat(trf).shape
        assert np.all(np.isfinite(trf.tvals_))
        assert np.all((trf.pvals_ > 0) & (trf.pvals_ <= 1))

        # closed-form cross-check: se = sqrt(diag(inv(XᵀX)) * SSE/dof).
        # tvals_ are stored in solve order, i.e. the lag-flip of coef_
        dof = len(Y) - X.shape[1]
        sigma = np.sum((Y - trf.predict(X)) ** 2, axis=0) / dof
        se_manual = np.sqrt(np.diag(np.linalg.inv(X.T @ X))[:, None] * sigma)
        t_manual = self._coef_flat(trf)[::-1] / se_manual
        assert np.allclose(trf.tvals_, t_manual)

    def test_trf_stats_with_intercept(self):
        """Unregularized fit with fit_intercept=True must compute t-/p-values
        with the intercept row stripped correctly."""
        trf, X, Y = self._fit_trf_stats(fit_intercept=True)
        assert trf.tvals_.shape == self._coef_flat(trf).shape
        assert trf.pvals_.shape == self._coef_flat(trf).shape
        assert np.all(np.isfinite(trf.tvals_))
        assert np.all((trf.pvals_ > 0) & (trf.pvals_ <= 1))

    def test_trf_stats_multi_channel(self):
        """Stats must work for multiple channels (se shape (n_coefs, n_chans))."""
        trf, X, Y = self._fit_trf_stats(fit_intercept=False, n_chans=2)
        assert trf.tvals_.shape == self._coef_flat(trf).shape
        assert np.all(np.isfinite(trf.tvals_))

    def test_trf_recovers_bipolar_kernel(self):
        """Bipolar (derivative-of-gaussian) kernel should also be recoverable."""
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, kernel_type='bipolar', seed=7)
        coef = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=0.001)

        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.95, f"Bipolar kernel correlation {corr:.3f} < 0.95"

    def test_trf_smooth_input(self):
        """TRFEstimator should work with smooth (non-pulse) stimulus."""
        srate = 100
        tker, ker = dummy_trf_kernel(srate=srate)
        t, x = simulate_smooth_input(dur=30.0, srate=srate, seed=42)
        y = simulate_trf_output(tker, ker, x, srate=srate)

        coef = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=0.001)
        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.85, f"Smooth input kernel correlation {corr:.3f} < 0.85"

    def test_trf_score_positive(self):
        """TRFEstimator.score should be positive for a recoverable kernel."""
        from pyeeg.models import TRFEstimator

        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, n_events=300, seed=42)

        lags = np.round(tker * srate).astype(int)
        X = lag_matrix(x[:, None], lags, filling=0., drop_missing=False)
        trf = TRFEstimator(times=tker, srate=srate, alpha=1.0, fit_intercept=False)
        trf.fit(X[:, ::-1], y[:, None], lagged=True, drop=False)
        r2 = trf.score(X[:, ::-1], y[:, None])

        assert r2 > 0.0, f"R² should be positive, got {r2:.3f}"

    def test_trf_tmin_tmax(self):
        """TRFEstimator with tmin/tmax should work and recover kernel."""
        from pyeeg.models import TRFEstimator

        srate = 100
        tker, ker = dummy_trf_kernel(srate=srate)
        t, x, y, _, _ = _make_trf_data(srate=srate, seed=42)

        lags = np.round(tker * srate).astype(int)
        X = lag_matrix(x[:, None], lags, filling=0., drop_missing=False)
        trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=srate, alpha=1.0, fit_intercept=False)
        trf.fit(X[:, ::-1], y[:, None], lagged=True, drop=False)

        assert trf.coef_ is not None
        assert trf.coef_.shape[0] > 0

    def test_trf_multi_channel(self):
        """TRFEstimator should handle multi-channel y."""
        from pyeeg.models import TRFEstimator

        srate = 100
        tker, ker = dummy_trf_kernel(srate=srate)
        t, x = simulate_pulse_inputs(n_events=200, dur=30.0, srate=srate, seed=42)
        rng = np.random.default_rng(99)
        y1 = simulate_trf_output(tker, ker, x, srate=srate)
        y2 = rng.standard_normal(len(y1)) * 0.1
        y2d = np.column_stack([y1, y2])

        lags = np.round(tker * srate).astype(int)
        X = lag_matrix(x[:, None], lags, filling=0., drop_missing=False)
        trf = TRFEstimator(times=tker, srate=srate, alpha=1.0, fit_intercept=False)
        trf.fit(X[:, ::-1], y2d, lagged=True, drop=False)

        coef = trf.coef_[:, 0, 0]  # first feature, first channel
        corr = np.corrcoef(coef, ker)[0, 1]
        assert corr > 0.90, f"Multi-channel kernel recovery {corr:.3f} < 0.90"

    def test_trf_alpha_increases_shrinkage(self):
        """Higher alpha should produce smaller coefficient norms (more shrinkage)."""
        srate = 100
        t, x, y, tker, ker = _make_trf_data(srate=srate, seed=42)

        coef_low = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=0.1)
        coef_high = _fit_trf_lagged(tker, ker, x, y, srate=srate, alpha=100.0)

        assert np.linalg.norm(coef_high) < np.linalg.norm(coef_low), \
            "Higher alpha should produce smaller coefficient norms"


# ===========================================================================
# CCA tests
# ===========================================================================

class TestCCAEstimator:
    """Regression tests for CCA_Estimator using simulated data."""

    def test_cca_recovers_correlations(self):
        """CCA should find high correlations in data with known shared structure."""
        from pyeeg.cca import cca_svd

        x, y, Z, A, B = _make_cca_data(n=5000, n_feats=3, n_chans=4, noise=0.3, seed=42)
        Ax, Ay, R = cca_svd(x, y)

        assert R[0] > 0.8, f"First canonical correlation {R[0]:.3f} < 0.8"

    def test_cca_decreasing_correlations(self):
        """Canonical correlations should be in decreasing order."""
        from pyeeg.cca import cca_svd

        x, y, Z, A, B = _make_cca_data(n=5000, n_feats=4, n_chans=4, noise=0.5, seed=42)
        Ax, Ay, R = cca_svd(x, y)

        for i in range(len(R) - 1):
            assert R[i] >= R[i + 1] - 1e-10, \
                f"Correlations not decreasing: R[{i}]={R[i]:.4f} < R[{i+1}]={R[i+1]:.4f}"

    def test_cca_projections_orthonormal(self):
        """CCA projection matrices should yield orthonormal canonical variables."""
        from pyeeg.cca import cca_svd

        x, y, Z, A, B = _make_cca_data(n=5000, n_feats=3, n_chans=3, noise=0.5, seed=42)
        Ax, Ay, R = cca_svd(x, y)
        px = x @ Ax
        py = y @ Ay

        Cx = np.corrcoef(px, rowvar=False)
        Cy = np.corrcoef(py, rowvar=False)

        np.testing.assert_allclose(Cx, np.eye(Cx.shape[0]), atol=0.05,
                                   err_msg="X canonical variables not approximately orthonormal")
        np.testing.assert_allclose(Cy, np.eye(Cy.shape[0]), atol=0.05,
                                   err_msg="Y canonical variables not approximately orthonormal")

    def test_cca_zero_signal_recovery(self):
        """CCA with perfectly shared signal should find correlation ~1."""
        from pyeeg.cca import cca_svd

        rng = np.random.default_rng(42)
        n = 2000
        shared = rng.standard_normal((n, 1))
        noise = 0.01
        x = np.hstack([shared + noise * rng.standard_normal((n, 1)),
                        rng.standard_normal((n, 2))])
        y = np.hstack([shared + noise * rng.standard_normal((n, 1)),
                        rng.standard_normal((n, 2))])

        Ax, Ay, R = cca_svd(x, y)
        assert R[0] > 0.99, f"Perfect signal correlation {R[0]:.4f} < 0.99"

    def test_cca_nt_vs_svd_agreement(self):
        """cca_nt and cca_svd should agree on canonical correlations."""
        from pyeeg.cca import cca_nt, cca_svd

        x, y, Z, A, B = _make_cca_data(n=3000, n_feats=3, n_chans=3, noise=0.5, seed=42)
        Ax, Ay, R_svd = cca_svd(x, y)
        A1, A2, A, B_coef, R_nt, _, _ = cca_nt(x, y, [1, 1], None)

        np.testing.assert_allclose(R_svd, R_nt, atol=0.01,
                                   err_msg="cca_svd and cca_nt disagree on correlations")
