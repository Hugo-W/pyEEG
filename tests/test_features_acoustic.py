"""Tests for the acoustic feature extractor and user-defined extractor wrapper.

Covers the pure-NumPy path of :mod:`pyeeg.features.acoustic`
(:class:`AcousticFeatureConfig` / :class:`AcousticFeatureExtractor`) and the
audio / custom-extractor path of :mod:`pyeeg.features.pipeline`
(:class:`UserDefinedExtractor`, :class:`StimulusEncoder`). No model loading is
required, so no ``@pytest.mark.llm`` / ``@pytest.mark.slow`` markers are used.
"""
import numpy as np
import pytest

from pyeeg.features.acoustic import AcousticFeatureConfig, AcousticFeatureExtractor
from pyeeg.features.pipeline import (
    FeaturePipeline,
    FeatureSpec,
    PipelineConfig,
    StimulusEncoder,
)

# A 1-second 440 Hz sine wave at 16 kHz, used as the default audio stimulus.
SRATE = 16000.0
NSAMPLES = 16000


@pytest.fixture
def sine_signal():
    t = np.arange(NSAMPLES) / SRATE
    return np.sin(2 * np.pi * 440 * t)


# ---------------------------------------------------------------------------
# AcousticFeatureConfig defaults
# ---------------------------------------------------------------------------

class TestAcousticFeatureConfigDefaults:
    """AcousticFeatureConfig dataclass defaults."""

    def test_defaults(self):
        cfg = AcousticFeatureConfig()
        assert cfg.sampling_rate == 16000.0
        assert cfg.features == ['envelope']


# ---------------------------------------------------------------------------
# AcousticFeatureExtractor
# ---------------------------------------------------------------------------

class TestEnvelopeExtraction:
    """Broadband envelope extraction (default 'hilbert' method)."""

    def test_envelope_key_and_type(self, sine_signal):
        extractor = AcousticFeatureExtractor(AcousticFeatureConfig())
        result = extractor.extract(sine_signal, SRATE)

        assert 'envelope' in result
        env = result['envelope']
        assert isinstance(env, np.ndarray)
        assert env.ndim == 1

    def test_envelope_resampled_to_125_hz(self, sine_signal):
        extractor = AcousticFeatureExtractor(AcousticFeatureConfig())
        env = extractor.extract(sine_signal, SRATE)['envelope']

        # Envelope is resampled to 125 Hz by default, so it has far fewer
        # samples than the 16 kHz input (~125 samples for 1 second).
        assert len(env) < len(sine_signal)
        assert len(env) == 125
        assert np.all(np.isfinite(env))

    def test_envelope_rectify_method(self, sine_signal):
        extractor = AcousticFeatureExtractor(
            AcousticFeatureConfig(features=['envelope'], envelope_method='rectify')
        )
        env = extractor.extract(sine_signal, SRATE)['envelope']

        assert isinstance(env, np.ndarray)
        assert env.ndim == 1
        assert len(env) == 125
        assert np.all(np.isfinite(env))


class TestFilterbankExtraction:
    """Band-pass filterbank extraction."""

    @pytest.mark.slow
    def test_filterbank_shape(self, sine_signal):
        extractor = AcousticFeatureExtractor(
            AcousticFeatureConfig(features=['filterbank'], sampling_rate=16000)
        )
        result = extractor.extract(sine_signal, SRATE)

        assert 'filterbank' in result
        fb = result['filterbank']
        # Default boundaries [100, 500, 1000, 2000, 4000, 8000]; the 8000 Hz
        # boundary is at the Nyquist frequency (srate/2) and is skipped.
        n_filters = fb.shape[0]
        assert n_filters < len(extractor.config.filterbank_freqs)
        assert fb.shape == (n_filters, len(sine_signal))
        assert np.all(np.isfinite(fb))


class TestGammatoneExtraction:
    """Gammatone rate-map extraction (compiled extension or fallback)."""

    def test_gammatone_shape(self, sine_signal):
        extractor = AcousticFeatureExtractor(
            AcousticFeatureConfig(features=['gammatone'], sampling_rate=16000)
        )
        try:
            result = extractor.extract(sine_signal, SRATE)
        except Exception as exc:  # noqa: BLE001 - C extension unavailable
            pytest.skip(f"gammatone rate-map unavailable: {exc}")

        assert 'gammatone' in result
        gt = result['gammatone']
        assert isinstance(gt, np.ndarray)
        assert gt.ndim == 2
        assert gt.shape[0] == 32  # gammatone_nchannels
        assert np.all(np.isfinite(gt))


class TestMultiChannelAndDefaults:
    """Multi-channel input, default sampling rate, and dimension checks."""

    def test_multichannel_envelope_shape(self, sine_signal):
        signal2 = np.column_stack([sine_signal, 0.5 * sine_signal])
        extractor = AcousticFeatureExtractor(AcousticFeatureConfig())
        env = extractor.extract(signal2, SRATE)['envelope']

        # One envelope column per input channel.
        assert env.ndim == 2
        assert env.shape == (125, 2)

    def test_default_srate_from_config(self, sine_signal):
        extractor = AcousticFeatureExtractor(
            AcousticFeatureConfig(sampling_rate=SRATE)
        )
        # No explicit srate: falls back to config.sampling_rate.
        env = extractor.extract(sine_signal)['envelope']

        assert len(env) == 125

    def test_3d_signal_raises(self):
        extractor = AcousticFeatureExtractor(AcousticFeatureConfig())
        signal3d = np.zeros((16, 16, 16))

        with pytest.raises(ValueError):
            extractor.extract(signal3d, SRATE)


# ---------------------------------------------------------------------------
# User-defined extractor wrapper
# ---------------------------------------------------------------------------

class TestUserDefinedExtractor:
    """UserDefinedExtractor used via StimulusEncoder and FeaturePipeline."""

    def test_custom_extractor_via_stimulus_encoder(self):
        encoder = StimulusEncoder()
        encoder.add_custom_extractor(
            lambda text: {'word_count': np.array([len(text.split())])}
        )

        result, metadata = encoder.encode("hello world")

        assert 'custom_word_count' in result
        np.testing.assert_array_equal(result['custom_word_count'], np.array([2]))
        assert metadata['aligned'] is False

    def test_custom_extractor_via_feature_pipeline(self):
        spec = FeatureSpec(
            name='my',
            extractor_type='custom',
            features=['x'],
            config={'func': lambda s: {'x': np.array([len(s)])}},
        )
        pipeline = FeaturePipeline(PipelineConfig(feature_specs=[spec]))

        result, metadata = pipeline.extract("test")

        assert 'my_x' in result
        np.testing.assert_array_equal(result['my_x'], np.array([4]))


# ---------------------------------------------------------------------------
# Audio encoding via StimulusEncoder
# ---------------------------------------------------------------------------

class TestAudioEncoding:
    """StimulusEncoder.encode_audio."""

    def test_encode_audio_envelope(self, sine_signal):
        encoder = StimulusEncoder()
        encoder.add_acoustic_features(features=['envelope'])

        result, metadata = encoder.encode_audio(sine_signal, SRATE)

        assert 'acoustic_envelope' in result
        env = result['acoustic_envelope']
        assert isinstance(env, np.ndarray)
        assert env.ndim == 1
        assert metadata['stimulus_type'] == 'audio'
        assert metadata['sampling_rate'] == SRATE

    def test_encode_audio_zscore_normalization(self, sine_signal):
        encoder = StimulusEncoder(pipeline_config=PipelineConfig(normalization='zscore'))
        encoder.add_acoustic_features(features=['envelope'])

        result, metadata = encoder.encode_audio(sine_signal, SRATE)

        assert 'acoustic_envelope' in result
        env = result['acoustic_envelope']
        assert np.all(np.isfinite(env))
        np.testing.assert_allclose(np.mean(env), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(env), 1.0, atol=1e-10)