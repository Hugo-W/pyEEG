# -*- coding: utf-8 -*-
"""Unit tests for pyeeg.utils module.

Covers: lag_matrix, lag_span, lag_sparse, design_lagmatrix, mem_check,
        fir_order, sigmoid, sigmoid_derivative, is_pos_def, poisson_onsets,
        poisson_onsets_fixed_N, shift_array.
"""
import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pyeeg.utils import (
    lag_matrix,
    lag_span,
    lag_sparse,
    design_lagmatrix,
    mem_check,
    fir_order,
    sigmoid,
    sigmoid_derivative,
    is_pos_def,
    poisson_onsets,
    poisson_onsets_fixed_N,
    shift_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_col_data():
    """6-sample, 2-feature data matching the docstring example."""
    return np.array([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]], dtype=float)


@pytest.fixture
def single_col_data():
    """6-sample, 1-feature data."""
    return np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(-1, 1)


# ===========================================================================
# lag_matrix
# ===========================================================================

class TestLagMatrix:
    """Tests for lag_matrix (the current, non-deprecated API)."""

    # --- basic shapes -------------------------------------------------------

    def test_single_feature_full_mode(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0, 1), mode='full')
        assert out.shape == (6, 2)

    def test_single_feature_valid_mode(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0, 1), mode='valid')
        # lag=1 drops the first row → 5 rows
        assert out.shape == (5, 2)

    def test_two_feature_full_mode(self, two_col_data):
        out = lag_matrix(two_col_data, lags=(0, 1), mode='full')
        assert out.shape == (6, 4)

    def test_two_feature_valid_mode(self, two_col_data):
        out = lag_matrix(two_col_data, lags=(0, 1), mode='valid')
        assert out.shape == (5, 4)

    def test_output_columns_equal_nfeats_times_nlags(self, two_col_data):
        out = lag_matrix(two_col_data, lags=(0, 1, 2), mode='full')
        assert out.shape[1] == 2 * 3  # 2 features × 3 lags

    # --- values from docstring examples ------------------------------------

    def test_docstring_example_lags_ordered_by_lag(self, two_col_data):
        """Reproduce the docstring example for default block_order='lags'."""
        expected = np.array([
            [2, 8, 1, 7, 0, 0],
            [3, 9, 2, 8, 0, 0],
            [4, 10, 3, 9, 1, 7],
            [5, 11, 4, 10, 2, 8],
            [6, 12, 5, 11, 3, 9],
            [0, 0, 6, 12, 4, 10],
        ], dtype=float)
        out = lag_matrix(two_col_data, lags=(-1, 0, 2), mode='full')
        assert_array_equal(out, expected)

    def test_docstring_example_block_order_features(self, two_col_data):
        """Reproduce the docstring example for block_order='features'."""
        expected = np.array([
            [2, 1, 0, 8, 7, 0],
            [3, 2, 0, 9, 8, 0],
            [4, 3, 1, 10, 9, 7],
            [5, 4, 2, 11, 10, 8],
            [6, 5, 3, 12, 11, 9],
            [0, 6, 4, 0, 12, 10],
        ], dtype=float)
        out = lag_matrix(two_col_data, lags=(-1, 0, 2), mode='full', block_order='features')
        assert_array_equal(out, expected)

    # --- fill_value ---------------------------------------------------------

    def test_fill_value_nan(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0, 1), mode='full', fill_value=np.nan)
        assert np.isnan(out[0, 1])  # lag=1, first row should be NaN

    def test_fill_value_default_zero(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0, 1), mode='full')
        # Default fill_value is 0.0
        assert out[0, 1] == 0.0

    def test_custom_fill_value(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0, 1), mode='full', fill_value=-999.0)
        assert out[0, 1] == -999.0

    # --- negative and positive lags ----------------------------------------

    def test_negative_lag_shifts_past(self):
        """lag=-1 is negated to +1 internally, shifting data forward."""
        x = np.array([10, 20, 30, 40], dtype=float).reshape(-1, 1)
        out = lag_matrix(x, lags=(-1,), mode='full')
        # lag=-1 → negated to +1 → buf[:n-1] = x[1:] → [20,30,40], last gets fill
        assert_array_equal(out[:3, 0], [20, 30, 40])
        assert out[3, 0] == 0.0  # fill

    def test_positive_lag_shifts_future(self):
        """lag=1 is negated to -1 internally, shifting data backward."""
        x = np.array([10, 20, 30, 40], dtype=float).reshape(-1, 1)
        out = lag_matrix(x, lags=(1,), mode='full')
        # lag=1 → negated to -1 → buf[1:] = x[:3] → first gets fill, rest [10,20,30]
        assert out[0, 0] == 0.0  # fill
        assert_array_equal(out[1:, 0], [10, 20, 30])

    def test_zero_lag_returns_original(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(0,), mode='full')
        assert_array_equal(out[:, 0], single_col_data[:, 0])

    # --- valid mode trims correctly ----------------------------------------

    def test_valid_mode_trims_both_ends_with_mixed_lags(self):
        x = np.arange(10, dtype=float).reshape(-1, 1)
        out = lag_matrix(x, lags=(-2, 0, 3), mode='valid')
        # min_lag_arr = min(2, 0, -3) = -3, max_lag_arr = max(2, 0, -3) = 2
        # start = max(0, -(-3)) = 3, end = 10 - 2 = 8
        assert out.shape[0] == 5  # rows 3..7

    # --- block_order validation --------------------------------------------

    def test_invalid_block_order_raises(self, two_col_data):
        with pytest.raises(ValueError, match="block_order"):
            lag_matrix(two_col_data, lags=(0, 1), block_order='invalid')

    # --- invalid mode -------------------------------------------------------

    def test_invalid_mode_raises(self, single_col_data):
        with pytest.raises(ValueError, match="mode"):
            lag_matrix(single_col_data, lags=(0, 1), mode='bad')

    # --- 1d input is auto-reshaped -----------------------------------------

    def test_1d_input_accepted(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = lag_matrix(x, lags=(0, 1), mode='full')
        assert out.shape == (4, 2)

    # --- single lag ---------------------------------------------------------

    def test_single_lag(self, single_col_data):
        out = lag_matrix(single_col_data, lags=(2,), mode='full')
        assert out.shape == (6, 1)

    # --- block_order doesn't matter for single feature ---------------------

    def test_block_order_irrelevant_for_single_feature(self, single_col_data):
        out_lags = lag_matrix(single_col_data, lags=(0, 1), block_order='lags')
        out_feats = lag_matrix(single_col_data, lags=(0, 1), block_order='features')
        assert_array_equal(out_lags, out_feats)

    # --- deprecated kwargs forwarding --------------------------------------

    def test_deprecated_filling_kwarg(self, single_col_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = lag_matrix(single_col_data, lag_samples=(0, 1), filling=-1.0)
            assert any("deprecated" in str(warning.message).lower() for warning in w)
        assert out[0, 1] == -1.0

    def test_deprecated_drop_missing_kwarg(self, single_col_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = lag_matrix(single_col_data, lag_samples=(0, 1), drop_missing=True)
            assert any("deprecated" in str(warning.message).lower() for warning in w)
        # drop_missing=True → mode='valid'
        assert out.shape[0] == 5

    def test_deprecated_lag_samples_kwarg(self, single_col_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = lag_matrix(single_col_data, lag_samples=(0, 1))
            assert any("deprecated" in str(warning.message).lower() for warning in w)
        assert out.shape == (6, 2)

    # --- column ordering with multiple features ----------------------------

    def test_lags_order_interleaves_features(self, two_col_data):
        """With block_order='lags' and 2 features, columns alternate per lag group."""
        out = lag_matrix(two_col_data, lags=(0, 1), mode='full', block_order='lags')
        # lag=0 block: col0=feat0@lag0, col1=feat1@lag0
        # lag=1 block: col2=feat0@lag1, col3=feat1@lag1
        assert_array_equal(out[:, 0], two_col_data[:, 0])  # feat0, lag0 = original
        assert_array_equal(out[:, 1], two_col_data[:, 1])  # feat1, lag0 = original

    def test_features_order_groups_by_feature(self, two_col_data):
        """With block_order='features', all lags for feat0 come first."""
        out = lag_matrix(two_col_data, lags=(0, 1), mode='full', block_order='features')
        # feat0 block: col0=feat0@lag0, col1=feat0@lag1
        # feat1 block: col2=feat1@lag0, col3=feat1@lag1
        assert_array_equal(out[:, 0], two_col_data[:, 0])  # feat0, lag0
        assert out.shape[1] == 4


# ===========================================================================
# lag_span
# ===========================================================================

class TestLagSpan:

    def test_basic_output(self):
        lags = lag_span(0, 0.5, srate=100)
        assert_array_equal(lags, np.arange(0, 50))

    def test_negative_tmin(self):
        lags = lag_span(-0.1, 0.3, srate=100)
        assert lags[0] == -10  # ceil(-0.1 * 100) = -10
        assert lags[-1] == 29  # ceil(0.3 * 100) = 30, arange excludes endpoint

    def test_default_srate(self):
        lags = lag_span(0, 1.0)
        # default srate=125
        assert len(lags) == 125

    def test_fractional_samples_ceil(self):
        """ceil ensures we include the boundary sample."""
        lags = lag_span(0.01, 0.02, srate=100)
        # ceil(0.01*100)=1, ceil(0.02*100)=2 → arange(1,2) = [1]
        assert_array_equal(lags, [1])

    def test_output_is_int_array(self):
        lags = lag_span(0, 0.1, srate=100)
        assert lags.dtype.kind == 'i'

    def test_empty_range(self):
        """When tmin == tmax, should return empty array."""
        lags = lag_span(0.5, 0.5, srate=100)
        assert len(lags) == 0


# ===========================================================================
# lag_sparse
# ===========================================================================

class TestLagSparse:

    def test_basic_output(self):
        lags = lag_sparse([0.0, 0.008, 0.016], srate=125)
        # ceil(0*125)=0, ceil(0.008*125)=ceil(1.0)=1, ceil(0.016*125)=ceil(2.0)=2
        assert_array_equal(lags, [0, 1, 2])

    def test_default_srate(self):
        lags = lag_sparse([0.0, 0.1])
        # srate=125: ceil(0)=0, ceil(12.5)=13
        assert_array_equal(lags, [0, 13])

    def test_negative_times(self):
        lags = lag_sparse([-0.01], srate=100)
        # ceil(-0.01 * 100) = ceil(-1.0) = -1
        assert lags[0] == -1

    def test_single_time(self):
        lags = lag_sparse([0.5], srate=100)
        assert_array_equal(lags, [50])

    def test_output_shape(self):
        lags = lag_sparse([0.01, 0.02, 0.03, 0.04], srate=100)
        assert len(lags) == 4

    def test_ceil_rounding(self):
        """Non-integer sample values should be ceil'd up."""
        lags = lag_sparse([0.005], srate=100)
        # 0.005 * 100 = 0.5, ceil(0.5) = 1
        assert lags[0] == 1


# ===========================================================================
# design_lagmatrix
# ===========================================================================

class TestDesignLagmatrix:

    def test_1d_input_single_lag(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = design_lagmatrix(x, nlags=1)
        # 1D input → time_axis forced to 1, squeeze(-1) with k=1 → (4, 1)
        assert out.shape == (4, 1)

    def test_1d_input_multiple_lags(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = design_lagmatrix(x, nlags=3)
        # (n-nlags, nlags) = (2, 3)
        assert out.shape == (2, 3)

    def test_2d_input_single_feature(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        out = design_lagmatrix(x, nlags=2)
        # k=1 → squeezed to (n-nlags, nlags) = (3, 2)
        assert out.shape == (3, 2)

    def test_2d_input_multi_feature(self):
        x = np.arange(20, dtype=float).reshape(10, 2)
        out = design_lagmatrix(x, nlags=3)
        # (10-3, 3, 2) = (7, 3, 2)
        assert out.shape == (7, 3, 2)

    def test_time_axis_transpose(self):
        """If time_axis=1, data is transposed first."""
        x = np.arange(10, dtype=float).reshape(1, 10)  # 1 feature, 10 timepoints
        out = design_lagmatrix(x, nlags=2, time_axis=1)
        assert out.shape == (8, 2)

    def test_values_correct_single_lag(self):
        x = np.array([10.0, 20.0, 30.0, 40.0])
        out = design_lagmatrix(x, nlags=1)
        # 1D → (n-nlags, 1). Roll by 1: [40,10,20,30], trim first nlags → [10,20,30]
        assert_array_equal(out.flatten(), [10.0, 20.0, 30.0])

    def test_no_lag_zero(self):
        """design_lagmatrix should not include lag 0 (AR model design)."""
        x = np.arange(10, dtype=float)
        out = design_lagmatrix(x, nlags=3)
        # First column should NOT be the original signal
        assert not np.array_equal(out[:, 0], x[:7])


# ===========================================================================
# mem_check
# ===========================================================================

class TestMemCheck:

    def test_returns_positive_number(self):
        result = mem_check()
        assert result > 0

    def test_units_gb(self):
        result = mem_check(units='gb')
        assert result > 0
        assert result < 10000  # sanity: not more than 10TB

    def test_units_mb(self):
        result_mb = mem_check(units='mb')
        result_gb = mem_check(units='gb')
        assert result_mb > result_gb  # MB should be ~1024x larger

    def test_units_kb(self):
        result_kb = mem_check(units='kb')
        result_mb = mem_check(units='mb')
        assert result_kb > result_mb

    def test_units_bytes(self):
        result_bytes = mem_check(units='bytes')
        result_kb = mem_check(units='kb')
        assert result_bytes > result_kb

    def test_case_insensitive(self):
        assert mem_check('GB') > 0
        assert mem_check('Gb') > 0


# ===========================================================================
# fir_order
# ===========================================================================

class TestFirOrder:

    def test_returns_positive_int(self):
        order = fir_order(10, 1000)
        assert isinstance(order, (int, np.integer))
        assert order > 0

    def test_returns_odd_order(self):
        """Fir order should always be odd (Type I FIR filter)."""
        for tbw in [5, 10, 20, 50]:
            for srate in [250, 500, 1000, 44100]:
                order = fir_order(tbw, srate)
                assert order % 2 == 1, f"order={order} for tbw={tbw}, srate={srate}"

    def test_with_ripples(self):
        order = fir_order(10, 1000, atten=60, ripples=1e-3)
        assert order > 0
        assert order % 2 == 1

    def test_higher_atten_gives_higher_order(self):
        low = fir_order(10, 1000, atten=40)
        high = fir_order(10, 1000, atten=80)
        assert high > low

    def test_narrower_tbw_gives_higher_order(self):
        wide = fir_order(50, 1000)
        narrow = fir_order(5, 1000)
        assert narrow > wide


# ===========================================================================
# sigmoid / sigmoid_derivative
# ===========================================================================

class TestSigmoid:

    def test_at_x0_returns_half_rmax(self):
        assert sigmoid(0) == pytest.approx(0.5)

    def test_large_positive_saturates(self):
        assert sigmoid(100) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_saturates_zero(self):
        assert sigmoid(-100) == pytest.approx(0.0, abs=1e-10)

    def test_custom_rmax(self):
        assert sigmoid(0, rmax=10) == pytest.approx(5.0)

    def test_custom_x0(self):
        """At x=x0, sigmoid should equal rmax/2."""
        assert sigmoid(5, x0=5) == pytest.approx(0.5)

    def test_beta_stretches(self):
        """Higher beta makes transition steeper."""
        narrow = sigmoid(1, beta=1)
        steep = sigmoid(1, beta=10)
        assert steep > narrow  # for x > x0, higher beta → closer to rmax


class TestSigmoidDerivative:

    def test_peak_at_x0(self):
        """Derivative is maximal at x=x0."""
        x = np.linspace(-10, 10, 1000)
        dy = sigmoid_derivative(x)
        assert np.argmax(dy) == pytest.approx(500, abs=2)

    def test_zero_at_extremes(self):
        assert sigmoid_derivative(100) == pytest.approx(0.0, abs=1e-10)
        assert sigmoid_derivative(-100) == pytest.approx(0.0, abs=1e-10)

    def test_peak_value(self):
        """At x0 with defaults, derivative = beta * 0.5 * 0.5 = 0.25."""
        assert sigmoid_derivative(0, beta=1) == pytest.approx(0.25)


# ===========================================================================
# is_pos_def
# ===========================================================================

class TestIsPosDef:

    def test_identity_is_pos_def(self):
        assert is_pos_def(np.eye(3)) is True

    def test_known_pos_def(self):
        A = np.array([[2, 1], [1, 2]])
        assert is_pos_def(A) is True

    def test_singular_is_not_pos_def(self):
        A = np.array([[1, 1], [1, 1]])  # rank-1, eigenvalue 0
        assert is_pos_def(A) is False

    def test_negative_eigenvalue_not_pos_def(self):
        A = np.array([[1, 2], [2, 1]])  # eigenvalues: 3, -1
        assert is_pos_def(A) is False

    def test_asymmetric_returns_false(self):
        A = np.array([[1, 2], [3, 1]])
        assert is_pos_def(A) is False


# ===========================================================================
# poisson_onsets / poisson_onsets_fixed_N
# ===========================================================================

class TestPoissonOnsets:

    def test_returns_array(self):
        onsets = poisson_onsets(5.0, 1.0, seed=42)
        assert isinstance(onsets, np.ndarray)

    def test_onsets_within_duration(self):
        onsets = poisson_onsets(10.0, 2.0, seed=42)
        assert np.all(onsets >= 0)
        assert np.all(onsets < 2.0)

    def test_sorted_output(self):
        onsets = poisson_onsets(5.0, 1.0, seed=42)
        assert np.all(np.diff(onsets) > 0)

    def test_reproducible_with_seed(self):
        a = poisson_onsets(5.0, 1.0, seed=123)
        b = poisson_onsets(5.0, 1.0, seed=123)
        assert_array_equal(a, b)

    def test_higher_rate_gives_more_events(self):
        low = poisson_onsets(1.0, 10.0, seed=42)
        high = poisson_onsets(10.0, 10.0, seed=42)
        assert len(high) > len(low)


class TestPoissonOnsetsFixedN:

    def test_returns_correct_count(self):
        onsets = poisson_onsets_fixed_N(50, dur=5.0, seed=42)
        assert len(onsets) == 50

    def test_within_duration(self):
        onsets = poisson_onsets_fixed_N(100, dur=1.0, seed=42)
        assert np.all(onsets >= 0)
        assert np.all(onsets <= 1.0)

    def test_sorted(self):
        onsets = poisson_onsets_fixed_N(200, dur=2.0, seed=42)
        assert np.all(np.diff(onsets) >= 0)

    def test_reproducible(self):
        a = poisson_onsets_fixed_N(50, dur=1.0, seed=99)
        b = poisson_onsets_fixed_N(50, dur=1.0, seed=99)
        assert_array_equal(a, b)


# ===========================================================================
# shift_array
# ===========================================================================

class TestShiftArray:

    def test_basic_shape(self):
        arr = np.arange(20, dtype=float)
        out = shift_array(arr, win=5, overlap=0, padding=False)
        # n_windows = n_samples - win + 1 = 16
        assert out.shape == (5, 16)

    def test_with_overlap(self):
        arr = np.arange(20, dtype=float)
        out = shift_array(arr, win=5, overlap=2, padding=False)
        # step = 5-2 = 3, n_windows = (20-5)//3 + 1 = 6
        assert out.shape[0] == 5

    def test_window_too_small_raises(self):
        arr = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="window size"):
            shift_array(arr, win=1, overlap=0)

    def test_window_larger_than_data_raises(self):
        arr = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="window size"):
            shift_array(arr, win=10, overlap=0)

    def test_negative_overlap_raises(self):
        arr = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="Overlap"):
            shift_array(arr, win=3, overlap=-1)

    def test_overlap_exceeds_window_raises(self):
        arr = np.arange(10, dtype=float)
        with pytest.raises(ValueError, match="Overlap"):
            shift_array(arr, win=3, overlap=5)

    def test_padding_not_implemented(self):
        arr = np.arange(10, dtype=float)
        with pytest.raises(NotImplementedError):
            shift_array(arr, win=3, overlap=0, padding=True)