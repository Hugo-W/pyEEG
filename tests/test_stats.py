"""Tests for pyeeg.stats — TRF statistical inference.

Deterministic / invariant tests for the permutation engine, standardisation
contract, offset validation, and p-value computation.  Monte-Carlo
calibration tests are marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from pyeeg.models import TRFEstimator
from pyeeg.simulate import (
    dummy_trf_kernel,
    simulate_smooth_input,
    simulate_trf_output,
)
from pyeeg.stats import (
    permutation_test_trf,
    PermutationResult,
    _valid_offsets,
    _circular_shift,
    _fade_edges,
    _estimate_fade_samples,
    _zscore_array,
    _zscore_segments,
    _plus_one_pvals,
    _max_stat,
    _get_window_samples,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SRATE = 100
TMIN = -0.2
TMAX = 0.5


def _make_trf_data(srate=SRATE, dur=30.0, seed=42, noise=0.1):
    """Build a TRF dataset with known kernel + optional noise."""
    tker, ker = dummy_trf_kernel(
        tmin=TMIN, tmax=TMAX, srate=srate, tloc=0.1, sigma=0.1,
    )
    _, x = simulate_smooth_input(dur=dur, srate=srate, seed=seed)
    y_clean = simulate_trf_output(tker, ker, x, srate=srate)
    rng = np.random.default_rng(seed + 1)
    y = y_clean[:, None] + noise * rng.standard_normal((len(y_clean), 1))
    x = x[:, None]
    return x, y, tker, ker


def _make_null_data(srate=SRATE, dur=30.0, seed=42):
    """Build data where y is independent of x (null hypothesis)."""
    rng = np.random.default_rng(seed)
    n = int(dur * srate)
    x = rng.standard_normal((n, 1))
    y = rng.standard_normal((n, 1))
    return x, y


# ---------------------------------------------------------------------------
# _valid_offsets
# ---------------------------------------------------------------------------

class TestValidOffsets:
    def test_excludes_window(self):
        """Valid offsets must exclude the lag window ± margin."""
        # lags from -0.2 to 0.5 s at 100 Hz: tmin_samp=-20, tmax_samp=49
        # (lag_span is half-open: arange(-20, 50) → max 49)
        tmin_samp, tmax_samp = -20, 49
        window_extent = tmax_samp - tmin_samp  # 69
        margin = window_extent
        n = 3000
        offsets = _valid_offsets(n, tmin_samp, tmax_samp, margin=margin)

        # Invalid range: [-20 - 69, 49 + 69] = [-89, 118]
        for s in range(-89, 119):
            assert (s % n) not in offsets, (
                f"Offset {s % n} (from invalid s={s}) should not be in valid set."
            )

    def test_too_short_raises(self):
        """Segments too short for the lag window must raise."""
        with pytest.raises(ValueError, match="too short"):
            _valid_offsets(100, -20, 49)

    def test_returns_array(self):
        offsets = _valid_offsets(3000, -20, 49)
        assert isinstance(offsets, np.ndarray)
        assert offsets.dtype.kind == "i"
        assert len(offsets) > 0


# ---------------------------------------------------------------------------
# _circular_shift
# ---------------------------------------------------------------------------

class TestCircularShift:
    def test_shift_preserves_values(self):
        x = np.arange(10).reshape(-1, 1).astype(float)
        shifted = _circular_shift(x, 3)
        assert set(shifted.ravel()) == set(x.ravel())

    def test_shift_by_zero_is_identity(self):
        x = np.random.default_rng(0).standard_normal((20, 2))
        assert np.array_equal(_circular_shift(x, 0), x)

    def test_shift_is_reversible(self):
        x = np.random.default_rng(1).standard_normal((50, 3))
        assert np.array_equal(
            _circular_shift(_circular_shift(x, 7), -7), x
        )


# ---------------------------------------------------------------------------
# _fade_edges / _estimate_fade_samples
# ---------------------------------------------------------------------------

class TestFadeEdges:
    def test_fade_zeros_edges(self):
        """Fade should taper both edges to zero."""
        x = np.ones((100, 1))
        faded = _fade_edges(x, fade_samples=10)
        assert faded[0, 0] == 0.0
        assert faded[-1, 0] == 0.0
        # Middle should be ~1
        assert abs(faded[50, 0] - 1.0) < 1e-10

    def test_fade_preserves_shape(self):
        x = np.random.default_rng(0).standard_normal((50, 3))
        faded = _fade_edges(x, fade_samples=5)
        assert faded.shape == x.shape

    def test_fade_zero_is_copy(self):
        x = np.random.default_rng(0).standard_normal((20, 1))
        faded = _fade_edges(x, fade_samples=0)
        assert np.array_equal(faded, x)
        # Must be a copy, not the same array
        assert faded is not x

    def test_circular_shift_with_fade(self):
        """Circular shift with fade should produce smooth wrap-around."""
        x = np.ones((100, 1))
        shifted = _circular_shift(x, offset=50, fade_samples=10)
        # The wrap-around point (index 50) should be smooth (both sides near 0)
        assert abs(shifted[50, 0]) < 0.1  # faded edge meets faded edge

    def test_estimate_fade_white_noise(self):
        """White noise (flat spectrum, full bandwidth) → minimal fade."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, 1))
        fs = _estimate_fade_samples(x, srate=100)
        # White noise has full bandwidth → fade ≈ 1-2 samples
        assert fs <= 5, f"White noise fade should be minimal, got {fs}"

    def test_estimate_fade_smooth(self):
        """Smooth (low-pass) input → larger fade."""
        _, x = simulate_smooth_input(dur=20.0, srate=100, seed=42, fmax=10)
        x = x[:, None]
        fs = _estimate_fade_samples(x, srate=100)
        # Low-pass signal with ~10 Hz cutoff → fade should be several samples
        assert fs >= 5, f"Smooth input fade should be > 5, got {fs}"
        assert fs <= 500  # but not absurdly large

    def test_estimate_fade_capped(self):
        """Fade should be capped at max_fade_frac * n_samples."""
        _, x = simulate_smooth_input(dur=2.0, srate=100, seed=42, fmax=5)
        x = x[:, None]
        fs = _estimate_fade_samples(x, srate=100, max_fade_frac=0.1)
        assert fs <= 20  # 200 samples * 0.1 = 20


# ---------------------------------------------------------------------------
# _zscore_array / _zscore_segments
# ---------------------------------------------------------------------------

class TestZscore:
    def test_basic_zscore(self):
        arr = np.random.default_rng(0).standard_normal((100, 3))
        arr[:, 0] *= 10  # different scale
        z, zero_var = _zscore_array(arr)
        assert z.shape == arr.shape
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-10)
        assert not zero_var.any()

    def test_zero_variance(self):
        arr = np.ones((50, 2))
        arr[:, 1] = np.arange(50.0)
        z, zero_var = _zscore_array(arr)
        assert zero_var[0]  # constant column
        assert not zero_var[1]
        assert np.all(z[:, 0] == 0)  # zero-variance → zeros

    def test_weighted_zscore(self):
        rng = np.random.default_rng(2)
        arr = rng.standard_normal((100, 2))
        w = rng.uniform(0.1, 1.0, 100)
        z, _ = _zscore_array(arr, weights=w)
        # Weighted mean should be ~0
        w_mean = (w[:, None] * z).sum(axis=0) / w.sum()
        np.testing.assert_allclose(w_mean, 0, atol=1e-10)

    def test_per_segment_independence(self):
        """Different segments with different scales → per-segment z-scoring."""
        X_list = [
            np.random.default_rng(0).standard_normal((100, 1)) * 100,
            np.random.default_rng(1).standard_normal((100, 1)) * 0.01,
        ]
        y_list = [
            np.random.default_rng(2).standard_normal((100, 1)) * 50,
            np.random.default_rng(3).standard_normal((100, 1)) * 0.5,
        ]
        X_z, y_z, zvf, zvc = _zscore_segments(X_list, y_list)
        # Each segment should be individually standardised
        for Xi_z in X_z:
            np.testing.assert_allclose(Xi_z.mean(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(Xi_z.std(axis=0), 1, atol=1e-10)
        for yi_z in y_z:
            np.testing.assert_allclose(yi_z.mean(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(yi_z.std(axis=0), 1, atol=1e-10)
        assert not zvf.any()
        assert not zvc.any()


# ---------------------------------------------------------------------------
# _plus_one_pvals / _max_stat
# ---------------------------------------------------------------------------

class TestPvalsAndMaxStat:
    def test_plus_one_formula(self):
        """p = (1 + #{null >= obs}) / (1 + n_perm)."""
        obs = np.array([[[2.0]]])  # shape (1,1,1)
        null_max = np.array([1.0, 2.0, 3.0])
        family_mask = np.ones_like(obs, dtype=bool)
        pvals = _plus_one_pvals(obs, null_max, "two-sided", family_mask)
        # null_abs = [1, 2, 3]; obs_abs = 2
        # #{null >= 2} = 2 (values 2 and 3)
        # p = (1 + 2) / (1 + 3) = 0.75
        np.testing.assert_allclose(pvals, 0.75)

    def test_min_p_value(self):
        """Minimum attainable p = 1/(n_perm+1)."""
        obs = np.array([[[100.0]]])
        null_max = np.array([0.0, 0.0, 0.0, 0.0])
        family_mask = np.ones_like(obs, dtype=bool)
        pvals = _plus_one_pvals(obs, null_max, "two-sided", family_mask)
        np.testing.assert_allclose(pvals, 1 / (4 + 1))

    def test_max_stat_two_sided(self):
        stat_map = np.array([[[1.0, -3.0], [0.5, 0.0]]])
        family_mask = np.ones_like(stat_map, dtype=bool)
        ms = _max_stat(stat_map, "two-sided", family_mask)
        assert ms == 3.0

    def test_max_stat_positive(self):
        stat_map = np.array([[[1.0, -3.0], [0.5, 0.0]]])
        family_mask = np.ones_like(stat_map, dtype=bool)
        ms = _max_stat(stat_map, "positive", family_mask)
        assert ms == 1.0

    def test_max_stat_negative(self):
        stat_map = np.array([[[1.0, -3.0], [0.5, 0.0]]])
        family_mask = np.ones_like(stat_map, dtype=bool)
        ms = _max_stat(stat_map, "negative", family_mask)
        assert ms == -3.0

    def test_family_mask(self):
        """Unmasked elements get p=1.0."""
        obs = np.array([[[1.0, 100.0]]])
        null_max = np.array([0.5, 0.5, 0.5])
        family_mask = np.array([[[True, False]]])
        pvals = _plus_one_pvals(obs, null_max, "two-sided", family_mask)
        # Masked element: obs=1, null=[0.5,0.5,0.5] → #{null>=1}=0 → p=1/4
        np.testing.assert_allclose(pvals[0, 0, 0], 0.25)
        # Unmasked element → p=1.0
        np.testing.assert_allclose(pvals[0, 0, 1], 1.0)


# ---------------------------------------------------------------------------
# _get_window_samples
# ---------------------------------------------------------------------------

class TestGetWindowSamples:
    def test_from_tmin_tmax(self):
        trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=100)
        trf.fill_lags()
        tmin_samp, tmax_samp = _get_window_samples(trf)
        assert tmin_samp == -20
        assert tmax_samp == 49  # lag_span is half-open: arange(-20, 50) → max 49


# ---------------------------------------------------------------------------
# permutation_test_trf — deterministic tests
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_shape_parity_with_coef(self):
        """observed and pvals_corrected must match coef_ shape."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        result = permutation_test_trf(trf, x, y, n_perm=10, seed=42)
        assert result.observed.shape == trf.coef_.shape
        assert result.pvals_corrected.shape == trf.coef_.shape
        assert result.null_distribution.shape == (10,)

    def test_pvals_in_range(self):
        """All p-values must be in (0, 1]."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        result = permutation_test_trf(trf, x, y, n_perm=20, seed=42)
        pvals = result.pvals_corrected
        assert np.all(pvals > 0), "p-values must be > 0 (plus-one formula)"
        assert np.all(pvals <= 1), "p-values must be <= 1"

    def test_reproducibility(self):
        """Same seed → identical results."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        r1 = permutation_test_trf(trf, x, y, n_perm=10, seed=42)
        r2 = permutation_test_trf(trf, x, y, n_perm=10, seed=42)
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
        np.testing.assert_allclose(r1.pvals_corrected, r2.pvals_corrected)

    def test_reproducibility_n_jobs(self):
        """Same seed → identical results across n_jobs (within float tolerance)."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        r1 = permutation_test_trf(trf, x, y, n_perm=10, seed=42, n_jobs=1)
        r2 = permutation_test_trf(trf, x, y, n_perm=10, seed=42, n_jobs=2)
        # Tiny float differences (~1e-15) can arise from parallel execution order
        np.testing.assert_allclose(r1.null_distribution, r2.null_distribution,
                                   atol=1e-12)

    def test_stat_zscore_equals_manual(self):
        """stat='zscore' should equal manual z-score + refit."""
        x, y, _, _ = _make_trf_data()
        # Manual z-score
        x_z = (x - x.mean(axis=0)) / x.std(axis=0)
        y_z = (y - y.mean(axis=0)) / y.std(axis=0)
        trf_manual = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0,
                                  fit_intercept=False)
        trf_manual.fit(x_z, y_z)
        # Permutation test with zscore
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        result = permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="zscore")
        np.testing.assert_allclose(result.observed, trf_manual.coef_, atol=1e-10)

    def test_stat_zscore_invariance_to_scale(self):
        """Z-scored stat should be invariant to feature unit changes."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        r1 = permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="zscore")
        # Scale X by 1000
        r2 = permutation_test_trf(trf, x * 1000, y, n_perm=5, seed=42, stat="zscore")
        # Small float differences from SVD with different input scales
        np.testing.assert_allclose(r1.observed, r2.observed, atol=1e-8)

    def test_stat_zscore_fit_intercept_false(self):
        """Internal z-score refit should set fit_intercept=False."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0,
                          fit_intercept=True)
        result = permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="zscore")
        # The observed should match a fit with fit_intercept=False on z-scored data
        x_z = (x - x.mean(axis=0)) / x.std(axis=0)
        y_z = (y - y.mean(axis=0)) / y.std(axis=0)
        trf_noint = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0,
                                 fit_intercept=False)
        trf_noint.fit(x_z, y_z)
        np.testing.assert_allclose(result.observed, trf_noint.coef_, atol=1e-10)

    def test_stat_coef_uses_original_config(self):
        """stat='coef' should use the user's original TRF config."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        result = permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="coef")
        np.testing.assert_allclose(result.observed, trf.coef_, atol=1e-10)

    def test_stat_t_ols_only(self):
        """stat='t' should work for OLS and produce tvals-shaped output."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=None)
        trf.fit(x, y)
        result = permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="t")
        assert result.observed.shape == trf.coef_.shape
        # Should match tvals_ reshaped
        n_lags = len(trf.lags)
        n_feats = trf.n_feats_
        n_chans = trf.n_chans_
        np.testing.assert_allclose(
            result.observed,
            trf.tvals_.reshape(n_lags, n_feats, n_chans),
            atol=1e-10,
        )

    def test_stat_t_rejects_ridge(self):
        """stat='t' must raise for ridge (alpha > 0)."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        with pytest.raises(ValueError, match="requires OLS"):
            permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="t")

    def test_lagged_rejected_for_zscore(self):
        """lagged=True must be rejected for stat='zscore'."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        with pytest.raises(ValueError, match="raw.*pre-lag"):
            permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="zscore",
                                lagged=True)

    def test_feature_alphas_supported(self):
        """Banded ridge (feature_alphas) should work with stat='zscore'."""
        x, y, _, _ = _make_trf_data()
        # 2 features for feature_alphas
        x2 = np.hstack([x, np.random.default_rng(99).standard_normal(x.shape)])
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE,
                           feature_alphas=np.array([1.0, 10.0]))
        result = permutation_test_trf(trf, x2, y, n_perm=5, seed=42, stat="zscore")
        assert result.observed.ndim == 3
        assert result.observed.shape[1] == 2  # n_feats

    def test_robust_rejected_without_opt_in(self):
        """Robust loss must raise without allow_robust=True."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=None,
                           loss="cauchy")
        trf.fit(x, y)
        with pytest.raises(ValueError, match="allow_robust"):
            permutation_test_trf(trf, x, y, n_perm=5, seed=42, stat="zscore")

    def test_result_metadata(self):
        """Result should have hypothesis, resampling_scheme, stat, etc."""
        x, y, _, _ = _make_trf_data()
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        trf.fit(x, y)
        result = permutation_test_trf(trf, x, y, n_perm=5, seed=42)
        assert result.stat == "zscore"
        assert result.tails == "two-sided"
        assert result.n_perm == 5
        assert result.seed == 42
        assert "H0" in result.hypothesis
        assert "circular" in result.resampling_scheme
        assert result.zero_var_features is not None
        assert result.zero_var_channels is not None

    def test_multi_segment(self):
        """Multi-segment input should work (list of arrays)."""
        x1, y1, _, _ = _make_trf_data(dur=15, seed=10)
        x2, y2, _, _ = _make_trf_data(dur=20, seed=20)
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        result = permutation_test_trf(
            trf, [x1, x2], [y1, y2], n_perm=5, seed=42, stat="zscore"
        )
        assert result.observed.ndim == 3

    def test_zero_variance_feature_masked(self):
        """Zero-variance feature should be detected and masked."""
        x, y, _, _ = _make_trf_data()
        x_zero = x.copy()
        x_zero[:, 0] = 1.0  # constant feature
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
        result = permutation_test_trf(trf, x_zero, y, n_perm=5, seed=42,
                                      stat="zscore")
        assert result.zero_var_features[0]  # detected
        # The observed coef for the zero-var feature should be 0
        assert np.all(result.observed[:, 0, :] == 0)


# ---------------------------------------------------------------------------
# Slow Monte-Carlo calibration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPermutationCalibration:
    """Slow Monte-Carlo calibration tests for type-I error and power."""

    def test_type_I_control(self):
        """Under null (y independent of x), rejection rate ≤ ~6% at α=0.05."""
        n_sims = 50
        n_perm = 100
        rejections = 0
        for sim_seed in range(n_sims):
            x, y = _make_null_data(seed=sim_seed)
            trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=5.0)
            result = permutation_test_trf(
                trf, x, y, n_perm=n_perm, seed=sim_seed, stat="zscore"
            )
            if np.any(result.pvals_corrected < 0.05):
                rejections += 1
        rate = rejections / n_sims
        assert rate < 0.12, f"Type-I rate {rate:.2f} > 0.12 (expected < ~6%)"

    def test_power(self):
        """With a true TRF, rejection rate should be high."""
        x, y, _, _ = _make_trf_data(dur=60, noise=0.05)
        # Use OLS (alpha=None) for the power test — ridge with alpha=5.0 on
        # standardized data shrinks coefficients too much for the signal to
        # exceed the null at n_perm=100.
        trf = TRFEstimator(tmin=TMIN, tmax=TMAX, srate=SRATE, alpha=None)
        result = permutation_test_trf(
            trf, x, y, n_perm=100, seed=42, stat="zscore"
        )
        assert np.any(result.pvals_corrected < 0.05), (
            "Expected significant coefficients with a true TRF present."
        )
