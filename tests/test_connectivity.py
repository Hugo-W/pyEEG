# -*- coding: utf-8 -*-
"""Unit tests for pyeeg.connectivity module.

Covers: granger_causality, phase_transfer_entropy, jackknife_resample,
        csd_ndarray, wPLI, plm.

All tests use synthetic data with fixed random seeds for reproducibility.
"""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.signal import hilbert

# Compatibility shim: np.complex was removed in NumPy >= 2.0.
# pyeeg.connectivity.csd_ndarray uses np.complex; without this shim the
# import-time attribute access inside csd_ndarray would raise AttributeError.
if not hasattr(np, 'complex'):
    np.complex = complex

from pyeeg.connectivity import (
    granger_causality,
    phase_transfer_entropy,
    jackknife_resample,
    csd_ndarray,
    wPLI,
    plm,
)
from pyeeg.simulate import simulate_ar, simulate_var


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def coupled_xy():
    """Two-channel VAR(1) data with X → Y coupling.

    Model:  x_t = 0.5·x_{t-1} + ε₁
            y_t = 0.5·x_{t-1} + 0.5·y_{t-1} + ε₂

    Channel 0 (X) drives channel 1 (Y) but not vice-versa.
    """
    coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])  # shape (1, 2, 2)
    return simulate_var(1, coef, nobs=2000, ndim=2, seed=42)


@pytest.fixture
def independent_xy():
    """Two independent AR(1) processes."""
    x = simulate_ar(1, [0.5], 2000, seed=42)
    y = simulate_ar(1, [0.5], 2000, seed=43)
    return np.column_stack([x, y])


@pytest.fixture
def sinusoid_pair():
    """Two sinusoids at 5 Hz with a 90° phase lag (via Hilbert transform).

    Returns (data, fs) where data has shape (1000, 2) and fs=100.
    """
    fs = 100.0
    t = np.linspace(0, 10, 1000)
    sig = np.sin(2 * np.pi * 5 * t)
    sig_shifted = np.imag(hilbert(sig))  # 90° phase-shifted
    return np.column_stack([sig, sig_shifted]), fs


# ===========================================================================
# granger_causality
# ===========================================================================

class TestGrangerCausality:
    """Tests for granger_causality."""

    def test_output_shape(self, coupled_xy):
        GC = granger_causality(coupled_xy, nlags=1)
        assert GC.shape == (2, 2)

    def test_diagonal_is_zero(self, coupled_xy):
        GC = granger_causality(coupled_xy, nlags=1)
        assert_array_equal(np.diag(GC), [0.0, 0.0])

    def test_causal_direction_higher(self, coupled_xy):
        """X → Y coupling: GC[0, 1] (X causes Y) >> GC[1, 0] (Y causes X)."""
        GC = granger_causality(coupled_xy, nlags=1)
        assert GC[0, 1] > 0.1  # substantial causality X → Y
        assert GC[1, 0] < 0.01  # negligible causality Y → X
        assert GC[0, 1] > GC[1, 0]

    def test_matrix_is_asymmetric(self, coupled_xy):
        """GC matrix should be asymmetric when coupling is directional."""
        GC = granger_causality(coupled_xy, nlags=1)
        assert not np.allclose(GC, GC.T)

    def test_independent_signals_near_zero(self, independent_xy):
        """Independent AR processes should yield near-zero GC in both directions."""
        GC = granger_causality(independent_xy, nlags=1)
        assert_allclose(GC, np.zeros((2, 2)), atol=0.01)

    def test_time_axis_1_matches_transposed(self, coupled_xy):
        """time_axis=1 on transposed data should give the same result."""
        GC_t0 = granger_causality(coupled_xy, nlags=1, time_axis=0)
        GC_t1 = granger_causality(coupled_xy.T, nlags=1, time_axis=1)
        assert_allclose(GC_t0, GC_t1)

    def test_nlags_2(self):
        """GC should work with higher model order."""
        coef = np.array([
            [[0.5, 0.3], [0.0, 0.4]],
            [[0.1, 0.0], [0.0, 0.1]],
        ])
        data = simulate_var(2, coef, nobs=2000, ndim=2, seed=42)
        GC = granger_causality(data, nlags=2)
        assert GC.shape == (2, 2)
        assert GC[0, 1] > GC[1, 0]

    def test_three_channels(self):
        """GC should generalise to more than two channels."""
        coef = np.array([[
            [0.5, 0.3, 0.0],
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.5],
        ]])
        data = simulate_var(1, coef, nobs=2000, ndim=3, seed=42)
        GC = granger_causality(data, nlags=1)
        assert GC.shape == (3, 3)
        assert_array_equal(np.diag(GC), [0.0, 0.0, 0.0])
        # Channel 0 drives 1 and 2 (via 1), so GC[0, 1] should be largest
        assert GC[0, 1] > 0.05


# ===========================================================================
# wPLI
# ===========================================================================

class TestWPLI:
    """Tests for wPLI (weighted Phase Lag Index)."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self):
        """wPLI divides by zero when the imaginary CSD is zero (expected)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            yield

    def test_output_shape_with_fbands(self, rng):
        """With fbands, output is (nchannels, nchannels)."""
        x = rng.standard_normal((500, 3))
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert C.shape == (3, 3)

    def test_output_shape_full_spectrum(self, sinusoid_pair):
        """Without fbands, output is (nchannels, nchannels, nfreqs)."""
        data, fs = sinusoid_pair
        C = wPLI(data, fs=fs)
        # nfreqs = (nsamples-1)//2 + 1 after jackknife (999 samples → 500)
        assert C.shape[0] == 2
        assert C.shape[1] == 2
        assert C.shape[2] == 500

    def test_identical_signals_near_zero(self, rng):
        """Identical signals (zero phase lag) should give wPLI ≈ 0."""
        x = rng.standard_normal((500, 2))
        x[:, 1] = x[:, 0]
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert_allclose(C[0, 1], 0.0, atol=0.1)

    def test_independent_noise_near_zero(self, rng):
        """Independent white noise should give wPLI ≈ 0."""
        x = rng.standard_normal((500, 2))
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert abs(C[0, 1]) < 0.15

    def test_known_phase_lag_high(self, sinusoid_pair):
        """A consistent 90° phase lag should give |wPLI| ≈ 1."""
        data, fs = sinusoid_pair
        C = wPLI(data, fs=fs, fbands=(4.0, 6.0))
        assert abs(C[0, 1]) > 0.9

    def test_value_range(self, rng):
        """wPLI values should lie in [-1, 1]."""
        x = rng.standard_normal((500, 3))
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert np.all(np.abs(C) <= 1.0 + 1e-10)

    def test_3d_input_trials(self, rng):
        """3D input (trials, samples, channels) should be accepted."""
        x = rng.standard_normal((5, 200, 2))
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert C.shape == (2, 2)

    def test_symmetric(self, rng):
        """wPLI matrix should be symmetric (it is an undirected measure)."""
        x = rng.standard_normal((500, 3))
        C = wPLI(x, fs=1.0, fbands=(0.05, 0.4))
        assert_allclose(C, C.T)


# ===========================================================================
# plm
# ===========================================================================

class TestPLM:
    """Tests for plm (Phase Linearity Measurement)."""

    def test_output_shape(self, rng):
        x = rng.standard_normal((500, 3))
        C = plm(x, fband=0.1, fs=1.0)
        assert C.shape == (3, 3)

    def test_symmetric(self, rng):
        """PLM is an undirected measure: C should equal C.T."""
        x = rng.standard_normal((500, 3))
        C = plm(x, fband=0.1, fs=1.0)
        assert_allclose(C, C.T)

    def test_diagonal_is_zero(self, rng):
        x = rng.standard_normal((500, 3))
        C = plm(x, fband=0.1, fs=1.0)
        assert_array_equal(np.diag(C), [0.0, 0.0, 0.0])

    def test_value_range(self, rng):
        """PLM values should lie in [0, 1]."""
        x = rng.standard_normal((500, 4))
        C = plm(x, fband=0.1, fs=1.0)
        assert C.min() >= 0.0
        assert C.max() <= 1.0

    def test_identical_signals_near_one(self, rng):
        """Identical signals should give PLM ≈ 1 (all power at DC)."""
        x = rng.standard_normal((500, 2))
        x[:, 1] = x[:, 0]
        C = plm(x, fband=0.1, fs=1.0)
        assert C[0, 1] == pytest.approx(1.0, abs=0.05)

    def test_rowvar_transposes(self, rng):
        """rowvar=True should accept channels-first input."""
        x = rng.standard_normal((500, 3))
        C_default = plm(x, fband=0.1, fs=1.0)
        C_rowvar = plm(x.T, fband=0.1, fs=1.0, rowvar=True)
        assert_allclose(C_default, C_rowvar)

    def test_fband_assertion(self, rng):
        """fband >= Nyquist should raise an AssertionError."""
        x = rng.standard_normal((100, 2))
        with pytest.raises(AssertionError, match="fband"):
            plm(x, fband=1.0, fs=1.0)  # fband == fs/2, not < fs/2


# ===========================================================================
# csd_ndarray
# ===========================================================================

class TestCsdNdarray:
    """Tests for csd_ndarray (cross-spectral density)."""

    def test_output_shape_default_nfft(self, rng):
        """Default nfft = nsamples → nfreqs = N//2 + 1."""
        x = rng.standard_normal((500, 3))
        S = csd_ndarray(x, fs=1.0)
        assert S.shape == (3, 3, 251)

    def test_output_shape_custom_nfft(self, rng):
        """Custom nfft larger than N should give nfft//2 + 1 freq bins."""
        x = rng.standard_normal((500, 2))
        S = csd_ndarray(x, fs=1.0, nfft=1024)
        assert S.shape == (2, 2, 513)

    def test_complex_dtype(self, rng):
        x = rng.standard_normal((200, 2))
        S = csd_ndarray(x, fs=1.0)
        assert S.dtype == np.complex128

    def test_symmetric(self, rng):
        """S[i, j] should equal S[j, i] (the implementation copies the upper triangle)."""
        x = rng.standard_normal((200, 3))
        S = csd_ndarray(x, fs=1.0)
        for i in range(3):
            for j in range(i + 1, 3):
                assert_allclose(S[i, j], S[j, i])

    def test_diagonal_is_real(self, rng):
        """Auto-spectral density (diagonal) should be real-valued."""
        x = rng.standard_normal((200, 2))
        S = csd_ndarray(x, fs=1.0)
        assert_allclose(S[0, 0].imag, 0.0, atol=1e-10)
        assert_allclose(S[1, 1].imag, 0.0, atol=1e-10)

    def test_sinusoid_peak_frequency(self):
        """CSD of a sinusoid should peak at the sinusoid frequency."""
        fs = 100.0
        n = 1000
        t = np.arange(n) / fs
        freq = 10.0
        sig = np.sin(2 * np.pi * freq * t)
        data = np.column_stack([sig, sig])
        S = csd_ndarray(data, fs=fs, nfft=n)
        from scipy.fftpack import fftfreq
        freqs = fftfreq(n, d=1 / fs)[:n // 2 + 1]
        peak_idx = np.argmax(np.abs(S[0, 1]))
        assert freqs[peak_idx] == pytest.approx(freq, abs=fs / n)

    def test_two_channel_shape(self, rng):
        """Minimal two-channel case."""
        x = rng.standard_normal((100, 2))
        S = csd_ndarray(x, fs=1.0)
        assert S.shape == (2, 2, 51)


# ===========================================================================
# jackknife_resample
# ===========================================================================

class TestJackknifeResample:
    """Tests for jackknife_resample."""

    def test_output_shape_1d(self):
        x = np.arange(10.0)
        out = jackknife_resample(x)
        assert out.shape == (10, 9)

    def test_output_shape_2d(self):
        x = np.arange(20.0).reshape(10, 2)
        out = jackknife_resample(x)
        assert out.shape == (10, 9, 2)

    def test_output_shape_3d(self):
        x = np.zeros((10, 5, 3))
        out = jackknife_resample(x)
        assert out.shape == (10, 9, 5, 3)

    def test_n_resamples_equals_n_samples(self, rng):
        """Should produce exactly n_samples resamples."""
        x = rng.standard_normal((50, 3))
        out = jackknife_resample(x)
        assert out.shape[0] == 50

    def test_each_resample_length(self, rng):
        """Each resample should have n-1 observations."""
        x = rng.standard_normal((30, 2))
        out = jackknife_resample(x)
        assert out.shape[1] == 29

    def test_first_resample_removes_first_row(self, rng):
        x = rng.standard_normal((20, 3))
        out = jackknife_resample(x)
        assert_array_equal(out[0], x[1:])

    def test_last_resample_removes_last_row(self, rng):
        x = rng.standard_normal((20, 3))
        out = jackknife_resample(x)
        assert_array_equal(out[-1], x[:-1])

    def test_middle_resample(self, rng):
        """Resample i should equal data with row i removed."""
        x = rng.standard_normal((15, 2))
        out = jackknife_resample(x)
        i = 7
        assert_array_equal(out[i], np.delete(x, i, axis=0))

    def test_all_resamples_cover_all_exclusions(self, rng):
        """Every resample should differ from the original by exactly one row."""
        x = rng.standard_normal((10, 2))
        out = jackknife_resample(x)
        for i in range(10):
            assert_array_equal(out[i], np.delete(x, i, axis=0))

    def test_1d_values(self):
        """1D input: each resample is the original minus one element."""
        x = np.array([10.0, 20.0, 30.0, 40.0])
        out = jackknife_resample(x)
        assert_array_equal(out[0], [20.0, 30.0, 40.0])
        assert_array_equal(out[1], [10.0, 30.0, 40.0])
        assert_array_equal(out[2], [10.0, 20.0, 40.0])
        assert_array_equal(out[3], [10.0, 20.0, 30.0])


# ===========================================================================
# phase_transfer_entropy
# ===========================================================================

class TestPhaseTransferEntropy:
    """Tests for phase_transfer_entropy (PTE)."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self):
        """PTE generates RuntimeWarnings from empty histogram bins (expected)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            yield

    def test_output_shapes(self):
        """Both dPTE and PTE should have shape (nchannels, nchannels)."""
        coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])
        data = simulate_var(1, coef, nobs=300, ndim=2, seed=42)
        dPTE, PTE = phase_transfer_entropy(data, delay=1)
        assert dPTE.shape == (2, 2)
        assert PTE.shape == (2, 2)

    def test_diagonal_is_zero(self):
        """Self-PTE should be zero."""
        coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])
        data = simulate_var(1, coef, nobs=300, ndim=2, seed=42)
        dPTE, PTE = phase_transfer_entropy(data, delay=1)
        assert_array_equal(np.diag(dPTE), [0.0, 0.0])
        assert_array_equal(np.diag(PTE), [0.0, 0.0])

    def test_dpte_normalized(self):
        """dPTE[i, j] + dPTE[j, i] should equal 1 (normalised directional PTE)."""
        coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])
        data = simulate_var(1, coef, nobs=300, ndim=2, seed=42)
        dPTE, _ = phase_transfer_entropy(data, delay=1)
        assert dPTE[0, 1] + dPTE[1, 0] == pytest.approx(1.0)

    def test_coupled_direction(self):
        """For X → Y coupling, dPTE[0, 1] should be greater than dPTE[1, 0]."""
        coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])
        data = simulate_var(1, coef, nobs=300, ndim=2, seed=42)
        dPTE, _ = phase_transfer_entropy(data, delay=1)
        assert dPTE[0, 1] > dPTE[1, 0]

    def test_independent_signals(self):
        """Independent signals should produce valid (finite) PTE output."""
        x = simulate_ar(1, [0.5], 300, seed=42)
        y = simulate_ar(1, [0.5], 300, seed=43)
        data = np.column_stack([x, y])
        dPTE, PTE = phase_transfer_entropy(data, delay=1)
        assert dPTE.shape == (2, 2)
        assert PTE.shape == (2, 2)
        # Values should be finite (no NaN/inf from the computation)
        assert np.all(np.isfinite(dPTE))
        assert np.all(np.isfinite(PTE))

    def test_three_channels(self):
        """PTE should work with three channels."""
        coef = np.array([[
            [0.5, 0.3, 0.0],
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.5],
        ]])
        data = simulate_var(1, coef, nobs=300, ndim=3, seed=42)
        dPTE, PTE = phase_transfer_entropy(data, delay=1)
        assert dPTE.shape == (3, 3)
        assert PTE.shape == (3, 3)
        assert_array_equal(np.diag(dPTE), [0.0, 0.0, 0.0])
        # dPTE should be normalised for each pair
        for i in range(3):
            for j in range(i + 1, 3):
                assert dPTE[i, j] + dPTE[j, i] == pytest.approx(1.0)

    def test_explicit_delay(self):
        """Passing an explicit delay should skip the auto-estimation loop."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 2))
        dPTE, PTE = phase_transfer_entropy(data, delay=2)
        assert dPTE.shape == (2, 2)

    def test_reproducible(self):
        """Same input should give identical output (deterministic)."""
        coef = np.array([[[0.5, 0.5], [0.0, 0.5]]])
        data = simulate_var(1, coef, nobs=300, ndim=2, seed=42)
        dPTE1, PTE1 = phase_transfer_entropy(data, delay=1)
        dPTE2, PTE2 = phase_transfer_entropy(data, delay=1)
        assert_array_equal(dPTE1, dPTE2)
        assert_array_equal(PTE1, PTE2)
