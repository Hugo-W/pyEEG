"""Assertion-based tests for :func:`pyeeg.gammatone.gammatone_filter`.

``gammatone_filter`` is a thin NumPy wrapper around a C implementation
(``pyeeg.bin.gammatone_c``) with a ctypes fallback to a prebuilt shared
library (``gammatone_lib``).  These tests use deterministic sinusoids (no
random data) and validate the wrapper's contract:

* output shape, dtype and finiteness,
* a stronger basilar-membrane response on-frequency than off-frequency,
* a non-negative envelope that tracks the input amplitude,
* finite instantaneous phase and an instantaneous frequency near ``cf``
  in steady state,
* half-wave rectification (``hrect=1``),
* bit-exact reproducibility,
* parity between the wrapper and a direct call into whichever C backend
  loaded (extension module, or ctypes shared library as fallback).
"""

import ctypes
import importlib

import numpy as np
import pytest

import pyeeg.gammatone
from pyeeg.gammatone import gammatone_filter

# ---------------------------------------------------------------------------
# Deterministic test signal
# ---------------------------------------------------------------------------
FS = 4000  # sampling frequency (Hz)
DURATION = 1.0  # signal length (s)
CF = 1000.0  # centre frequency (Hz)
T = np.arange(0.0, DURATION, 1.0 / FS)
N_SAMPLES = T.size
# The gammatone filter has an onset transient, so response metrics are
# evaluated over the second half of the signal (steady state).
STEADY = slice(N_SAMPLES // 2, None)

OUTPUT_NAMES = ("bm", "env", "instp", "instf")


def sine(freq, amplitude=1.0, t=T):
    """Deterministic sinusoid of the given frequency and amplitude."""
    return amplitude * np.sin(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------
# The compiled extension (.so/.pyd) is invisible to static analysis, so probe
# for it dynamically; gammatone_filter uses it whenever it imports cleanly.
try:
    _gammatone_c_ext = importlib.import_module("pyeeg.bin.gammatone_c")
except ImportError:
    _gammatone_c_ext = None

# Only defined when the extension module failed to import and pyeeg.gammatone
# fell back to loading the prebuilt shared library via ctypes.
_gammatone_lib = getattr(pyeeg.gammatone, "gammatone_lib", None)

requires_c_extension = pytest.mark.skipif(
    _gammatone_c_ext is None,
    reason="pyeeg.bin.gammatone_c C extension is not available",
)
requires_ctypes_lib = pytest.mark.skipif(
    _gammatone_lib is None,
    reason="ctypes fallback library (gammatone_lib) is not available",
)


class TestOutputContract:
    """Shape, dtype and finiteness of the four returned signals."""

    def test_returns_four_arrays_matching_input_length(self):
        result = gammatone_filter(sine(CF), FS, CF)
        assert len(result) == 4
        for out in result:
            assert isinstance(out, np.ndarray)
            assert out.shape == (N_SAMPLES,)

    @pytest.mark.parametrize("n", [1, 2, 100, 400])
    def test_output_shape_for_various_lengths(self, n):
        x = sine(CF, t=np.arange(n) / FS)
        for out in gammatone_filter(x, FS, CF):
            assert out.shape == (n,)

    def test_output_dtype_is_float64(self):
        for out in gammatone_filter(sine(CF), FS, CF):
            assert out.dtype == np.float64

    def test_outputs_are_finite_for_clean_sinusoid(self):
        for out in gammatone_filter(sine(CF), FS, CF):
            assert np.all(np.isfinite(out))


class TestFrequencyResponse:
    """The filter must respond more strongly on-frequency than off-frequency."""

    def test_on_frequency_response_exceeds_off_frequency(self):
        bm_on, _, _, _ = gammatone_filter(sine(CF), FS, CF)
        bm_off, _, _, _ = gammatone_filter(sine(100.0), FS, CF)
        rms_on = np.sqrt(np.mean(bm_on[STEADY] ** 2))
        rms_off = np.sqrt(np.mean(bm_off[STEADY] ** 2))
        # A unit-amplitude tone at cf passes with substantial gain.
        assert rms_on > 0.5
        # A tone far below cf is strongly attenuated (observed ratio ~1600x).
        assert rms_off < 0.01
        assert rms_on > 100 * rms_off


class TestEnvelope:
    """The instantaneous envelope must be non-negative and track amplitude."""

    def test_envelope_is_nonnegative(self):
        _, env, _, _ = gammatone_filter(sine(CF), FS, CF)
        assert np.all(env >= 0.0)

    def test_envelope_is_flat_for_steady_tone(self):
        # For a pure tone the steady-state envelope is (nearly) constant.
        _, env, _, _ = gammatone_filter(sine(CF), FS, CF)
        steady = env[STEADY]
        assert np.ptp(steady) < 0.05 * np.median(steady)

    def test_envelope_magnitude_tracks_input_amplitude(self):
        # The gammatone gain at cf is ~1.24, so a unit-amplitude tone yields a
        # steady-state envelope of the same order of magnitude as the input.
        _, env, _, _ = gammatone_filter(sine(CF, amplitude=1.0), FS, CF)
        assert np.median(env[STEADY]) == pytest.approx(1.0, rel=0.5)

    def test_envelope_scales_linearly_with_input_amplitude(self):
        _, env_full, _, _ = gammatone_filter(sine(CF, amplitude=1.0), FS, CF)
        _, env_half, _, _ = gammatone_filter(sine(CF, amplitude=0.5), FS, CF)
        assert np.median(env_half[STEADY]) == pytest.approx(
            0.5 * np.median(env_full[STEADY]), rel=1e-9
        )


class TestInstantaneousPhaseFrequency:
    def test_instantaneous_phase_is_finite_and_bounded(self):
        _, _, instp, _ = gammatone_filter(sine(CF), FS, CF)
        assert np.all(np.isfinite(instp))
        # atan2-based phase kept continuous by unwrapping in the C code.
        assert np.all(np.abs(instp) <= np.pi + 1e-9)

    def test_instantaneous_frequency_is_finite(self):
        _, _, _, instf = gammatone_filter(sine(CF), FS, CF)
        assert np.all(np.isfinite(instf))

    def test_instantaneous_frequency_near_cf_in_steady_state(self):
        _, _, _, instf = gammatone_filter(sine(CF), FS, CF)
        steady = instf[STEADY]
        assert np.median(steady) == pytest.approx(CF, abs=1.0)
        # For a tone exactly at cf the phase is locked, so the deviation of
        # the instantaneous frequency from cf is tiny (observed < 1e-10).
        assert np.max(np.abs(steady - CF)) < 1.0


class TestHalfWaveRectification:
    """hrect=1 half-wave-rectifies the basilar membrane displacement."""

    def test_hrect_envelope_still_nonnegative_and_finite(self):
        _, env, _, _ = gammatone_filter(sine(CF), FS, CF, hrect=1)
        assert np.all(np.isfinite(env))
        assert np.all(env >= 0.0)

    def test_hrect_rectifies_bm(self):
        bm_full, _, _, _ = gammatone_filter(sine(CF), FS, CF, hrect=0)
        bm_rect, _, _, _ = gammatone_filter(sine(CF), FS, CF, hrect=1)
        # Negative excursions are clipped to zero, positive ones untouched.
        assert np.all(bm_rect >= 0.0)
        assert np.all(bm_rect[bm_full < 0] == 0.0)
        np.testing.assert_array_equal(bm_rect[bm_full >= 0], bm_full[bm_full >= 0])
        # The rectified response genuinely differs from the full response.
        assert not np.array_equal(bm_rect, bm_full)

    def test_hrect_leaves_env_phase_and_frequency_unchanged(self):
        # env/instp/instf are derived from the unrectified filter state, so
        # only bm changes when hrect is switched on (C source: the bm[t] < 0
        # clip is applied after the analytic-signal quantities are computed).
        out_full = gammatone_filter(sine(CF), FS, CF, hrect=0)
        out_rect = gammatone_filter(sine(CF), FS, CF, hrect=1)
        for name, full, rect in zip(OUTPUT_NAMES, out_full, out_rect, strict=True):
            if name == "bm":
                assert not np.array_equal(full, rect)
            else:
                np.testing.assert_array_equal(full, rect)


class TestReproducibility:
    @pytest.mark.parametrize("hrect", [0, 1])
    def test_same_input_gives_identical_output(self, hrect):
        x = sine(CF)
        first = gammatone_filter(x, FS, CF, hrect=hrect)
        second = gammatone_filter(x, FS, CF, hrect=hrect)
        for name, a, b in zip(OUTPUT_NAMES, first, second, strict=True):
            np.testing.assert_array_equal(a, b, err_msg=name)


class TestInputHandling:
    def test_accepts_array_like_input(self):
        x = sine(CF)
        from_array = gammatone_filter(x, FS, CF)
        from_list = gammatone_filter(list(x), FS, CF)
        assert all(out.dtype == np.float64 for out in from_list)
        for name, expected, actual in zip(
            OUTPUT_NAMES, from_array, from_list, strict=True
        ):
            np.testing.assert_array_equal(actual, expected, err_msg=name)

    def test_accepts_integer_signal(self):
        x = np.array([0, 1, 0, -1] * 250, dtype=np.int64)
        result = gammatone_filter(x, FS, CF)
        for out in result:
            assert out.dtype == np.float64
            assert out.shape == (len(x),)
            assert np.all(np.isfinite(out))

    def test_silence_gives_zero_bm_and_env(self):
        n = 1000
        bm, env, instp, instf = gammatone_filter(np.zeros(n), FS, CF)
        np.testing.assert_array_equal(bm, np.zeros(n))
        np.testing.assert_array_equal(env, np.zeros(n))
        assert np.all(np.isfinite(instp))
        assert np.all(np.isfinite(instf))


@requires_c_extension
class TestCExtensionParity:
    """The wrapper must match a direct call into the C extension module."""

    @staticmethod
    def _call_extension(x, fs, cf, hrect):
        if _gammatone_c_ext is None:
            pytest.skip("pyeeg.bin.gammatone_c C extension is not available")
        nsamples = len(x)
        bm = np.zeros(nsamples, dtype=np.float64)
        env = np.zeros(nsamples, dtype=np.float64)
        instp = np.zeros(nsamples, dtype=np.float64)
        instf = np.zeros(nsamples, dtype=np.float64)
        _gammatone_c_ext.gammatone_c(
            x, nsamples, fs, cf, hrect, bm, env, instp, instf
        )
        return bm, env, instp, instf

    @pytest.mark.parametrize("hrect", [0, 1])
    def test_wrapper_matches_direct_extension_call(self, hrect):
        x = sine(CF)
        expected = self._call_extension(x, FS, CF, hrect)
        actual = gammatone_filter(x, FS, CF, hrect=hrect)
        for name, exp, act in zip(OUTPUT_NAMES, expected, actual, strict=True):
            np.testing.assert_allclose(
                act, exp, rtol=1e-12, atol=1e-12, err_msg=name
            )


@requires_ctypes_lib
class TestCtypesFallbackParity:
    """The wrapper must match a direct call into the ctypes shared library.

    Only runs when the extension module is unavailable and pyeeg.gammatone
    fell back to loading the prebuilt shared library via ctypes.
    """

    @staticmethod
    def _call_ctypes_lib(x, fs, cf, hrect):
        nsamples = len(x)
        bm = np.zeros(nsamples, dtype=np.float64)
        env = np.zeros(nsamples, dtype=np.float64)
        instp = np.zeros(nsamples, dtype=np.float64)
        instf = np.zeros(nsamples, dtype=np.float64)
        # argtypes are configured on the library by pyeeg.gammatone at import.
        _gammatone_lib.gammatone_c(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nsamples,
            fs,
            cf,
            hrect,
            bm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            env.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            instp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            instf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return bm, env, instp, instf

    @pytest.mark.parametrize("hrect", [0, 1])
    def test_wrapper_matches_direct_ctypes_call(self, hrect):
        x = sine(CF)
        expected = self._call_ctypes_lib(x, FS, CF, hrect)
        actual = gammatone_filter(x, FS, CF, hrect=hrect)
        for name, exp, act in zip(OUTPUT_NAMES, expected, actual, strict=True):
            np.testing.assert_allclose(
                act, exp, rtol=1e-12, atol=1e-12, err_msg=name
            )
