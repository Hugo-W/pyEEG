"""Tests for the feature extraction pipeline (pyeeg.features.pipeline).

Covers the no-model path (PipelineConfig defaults, empty pipeline, feature
normalization, StimulusEncoder basics) and the LLM-backed path (FeatureSpec /
StimulusEncoder extraction, normalization, and TextGrid alignment).

LLM tests are marked ``@pytest.mark.llm`` and use the tiny local GPT-2 model
provided by the ``tiny_gpt2_path`` fixture in ``tests/conftest.py``. They are
skipped when ``torch`` / ``transformers`` are unavailable.
"""
import numpy as np
import pytest

from pyeeg.features.pipeline import (
    FeaturePipeline,
    FeatureSpec,
    PipelineConfig,
    StimulusEncoder,
)
from pyeeg.features.alignment import AlignmentHandler

# Short sample text and a Praat TextGrid (long format) shared with the rest of
# the test suite (see tests/conftest.py).
SAMPLE_TEXT = "The cat sat on the mat."

TEXTGRID_STRING = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 3
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 3
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "The"
        intervals [2]:
            xmin = 1
            xmax = 2
            text = "cat"
        intervals [3]:
            xmin = 2
            xmax = 3
            text = "sat"
"""


# ---------------------------------------------------------------------------
# No-model pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineConfigDefaults:
    """PipelineConfig dataclass defaults."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.feature_specs == []
        assert cfg.alignment_config is None
        assert cfg.normalization == "none"
        assert cfg.cache_features is True

    def test_explicit_values(self):
        specs = [FeatureSpec(name="x", extractor_type="llm")]
        cfg = PipelineConfig(
            feature_specs=specs,
            alignment_config={"sampling_rate": 500.0},
            normalization="zscore",
            cache_features=False,
        )
        assert cfg.feature_specs == specs
        assert cfg.alignment_config == {"sampling_rate": 500.0}
        assert cfg.normalization == "zscore"
        assert cfg.cache_features is False


class TestFeatureSpec:
    """FeatureSpec dataclass defaults."""

    def test_defaults(self):
        spec = FeatureSpec(name="llm", extractor_type="llm")
        assert spec.features == []
        assert spec.config is None


class TestEmptyPipeline:
    """FeaturePipeline with no configured extractors."""

    def test_extract_returns_empty(self):
        pipeline = FeaturePipeline(PipelineConfig())
        features, metadata = pipeline.extract(SAMPLE_TEXT)
        assert features == {}
        assert metadata["aligned"] is False
        assert metadata["text"] == SAMPLE_TEXT
        assert metadata["feature_specs"] == []


class TestNormalizeFeatures:
    """FeaturePipeline.normalize_features."""

    def _pipeline(self):
        return FeaturePipeline(PipelineConfig())

    def test_none_returns_input(self):
        pipeline = self._pipeline()
        features = {"a": np.array([1.0, 2.0, 3.0])}
        result = pipeline.normalize_features(features, "none")
        assert result is features

    def test_zscore(self):
        pipeline = self._pipeline()
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        features = {"a": arr}
        result = pipeline.normalize_features(features, "zscore")
        mean = np.mean(arr)
        std = np.std(arr)
        np.testing.assert_allclose(result["a"], (arr - mean) / std)

    def test_zscore_constant_array(self):
        pipeline = self._pipeline()
        arr = np.array([5.0, 5.0, 5.0])
        result = pipeline.normalize_features({"a": arr}, "zscore")
        np.testing.assert_allclose(result["a"], arr - np.mean(arr))
        np.testing.assert_allclose(result["a"], np.zeros_like(arr))

    def test_minmax(self):
        pipeline = self._pipeline()
        arr = np.array([1.0, 3.0, 6.0])
        features = {"a": arr}
        result = pipeline.normalize_features(features, "minmax")
        min_val = np.min(arr)
        max_val = np.max(arr)
        np.testing.assert_allclose(result["a"], (arr - min_val) / (max_val - min_val))

    def test_minmax_constant_array(self):
        pipeline = self._pipeline()
        arr = np.array([2.0, 2.0, 2.0])
        result = pipeline.normalize_features({"a": arr}, "minmax")
        np.testing.assert_allclose(result["a"], arr - np.min(arr))
        np.testing.assert_allclose(result["a"], np.zeros_like(arr))

    def test_unknown_method_returns_input(self):
        pipeline = self._pipeline()
        features = {"a": np.array([1.0, 2.0, 3.0])}
        result = pipeline.normalize_features(features, "bogus")
        np.testing.assert_array_equal(result["a"], features["a"])


class TestStimulusEncoderBasics:
    """StimulusEncoder without any configured extractors."""

    def test_default_construction(self):
        encoder = StimulusEncoder()
        assert encoder.pipeline.config.feature_specs == []
        assert encoder.pipeline.config.alignment_config is None

    def test_encode_without_specs_returns_empty(self):
        encoder = StimulusEncoder()
        features, metadata = encoder.encode(SAMPLE_TEXT)
        assert features == {}
        assert metadata["aligned"] is False

    def test_encode_to_array_without_specs_returns_empty(self):
        encoder = StimulusEncoder()
        array = encoder.encode_to_array(SAMPLE_TEXT)
        assert isinstance(array, np.ndarray)
        assert array.size == 0


# ---------------------------------------------------------------------------
# LLM pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.llm
class TestLLMPipeline:
    """FeaturePipeline with an LLM extractor (tiny local GPT-2)."""

    def _spec(self, tiny_gpt2_path, features=("surprisal",)):
        return FeatureSpec(
            name="llm",
            extractor_type="llm",
            features=list(features),
            config={"model_name": tiny_gpt2_path, "device": "cpu"},
        )

    def test_feature_pipeline_llm_extract(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        spec = self._spec(tiny_gpt2_path)
        pipeline = FeaturePipeline(PipelineConfig(feature_specs=[spec]))

        features, metadata = pipeline.extract(SAMPLE_TEXT)

        assert "llm_surprisal" in features
        assert isinstance(features["llm_surprisal"], np.ndarray)
        assert features["llm_surprisal"].ndim == 1
        assert metadata["aligned"] is False

    def test_stimulus_encoder_llm_encode(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        encoder = StimulusEncoder()
        encoder.add_llm_features(features=["surprisal"], model_name=tiny_gpt2_path)

        features, metadata = encoder.encode(SAMPLE_TEXT)

        assert "llm_surprisal" in features
        assert isinstance(features["llm_surprisal"], np.ndarray)
        assert features["llm_surprisal"].ndim == 1
        assert metadata["aligned"] is False

    def test_stimulus_encoder_llm_encode_to_array(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        encoder = StimulusEncoder()
        encoder.add_llm_features(features=["surprisal"], model_name=tiny_gpt2_path)

        array = encoder.encode_to_array(SAMPLE_TEXT)

        assert isinstance(array, np.ndarray)
        assert array.ndim == 2
        assert array.shape[1] == 1  # one column per feature

    def test_stimulus_encoder_zscore_normalization(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        config = PipelineConfig(normalization="zscore")
        encoder = StimulusEncoder(pipeline_config=config)
        encoder.add_llm_features(features=["surprisal"], model_name=tiny_gpt2_path)

        features, metadata = encoder.encode(SAMPLE_TEXT)

        assert "llm_surprisal" in features
        arr = features["llm_surprisal"]
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0
        # The LLM extractor returns float32 arrays; allow float32 round-off.
        np.testing.assert_allclose(np.mean(finite), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(finite), 1.0, atol=1e-7)

    def test_stimulus_encoder_alignment(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        encoder = StimulusEncoder()
        encoder.add_llm_features(features=["surprisal"], model_name=tiny_gpt2_path)
        encoder.set_alignment(sampling_rate=1000.0)

        textgrid = AlignmentHandler().load_textgrid_from_string(TEXTGRID_STRING)
        textgrid.end_time = 3.0

        features, metadata = encoder.encode(
            SAMPLE_TEXT, textgrid=textgrid, signal_length=3000
        )

        assert "llm_surprisal" in features
        arr = features["llm_surprisal"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3000,)
        assert metadata["aligned"] is True
        assert metadata["n_samples"] == 3000
        assert metadata["sampling_rate"] == 1000.0

    def test_add_syntactic_features_appends_spec(self):
        encoder = StimulusEncoder()
        encoder.add_syntactic_features(features=["depth"])
        specs = encoder.pipeline.config.feature_specs
        assert len(specs) == 1
        assert specs[0].name == "syntactic"
        assert specs[0].extractor_type == "syntactic"
        assert specs[0].features == ["depth"]
        assert specs[0].config == {"parser_name": "stanford"}


# ---------------------------------------------------------------------------
# Regression tests: NaN handling, empty alignment, and defaults
# ---------------------------------------------------------------------------

class TestFeatureSpecConfigNone:
    """FeatureSpec with ``config=None`` falls back to extractor defaults."""

    def test_syntactic_config_none_constructs(self):
        # ``config=None`` must not raise AttributeError: the pipeline uses
        # ``spec.config or {}`` and instantiates the syntactic extractor with
        # its default ParserConfig (no model download needed).
        spec = FeatureSpec(
            name="syn", extractor_type="syntactic", features=["depth"], config=None
        )
        pipeline = FeaturePipeline(PipelineConfig(feature_specs=[spec]))

        assert "syn" in pipeline._extractors


class TestEmptyAlignedPipeline:
    """StimulusEncoder with alignment but no feature specs."""

    def test_encode_aligned_without_specs_returns_empty(self):
        encoder = StimulusEncoder()
        encoder.set_alignment(sampling_rate=1000.0)

        features, metadata = encoder.encode("hello world")

        # No feature specs -> all_features is empty -> the aligned loop does
        # not execute; exercises the ``aligned = np.array([])`` initialization.
        assert features == {}
        assert metadata["aligned"] is True
        assert metadata["n_samples"] == 0


class TestNormalizeFeaturesNaN:
    """FeaturePipeline.normalize_features with NaN-containing arrays."""

    def _pipeline(self):
        return FeaturePipeline(PipelineConfig())

    def test_zscore_all_nan_returns_all_nan(self):
        pipeline = self._pipeline()
        features = {"bad": np.array([np.nan, np.nan, np.nan])}

        # np.nanmean/np.nanstd on an all-NaN slice emit RuntimeWarnings; the
        # normalization must still return an all-NaN array instead of crashing.
        with pytest.warns(RuntimeWarning):
            result = pipeline.normalize_features(features, method="zscore")

        assert np.all(np.isnan(result["bad"]))

    def test_zscore_mixed_nan(self):
        pipeline = self._pipeline()
        features = {
            "good": np.array([1.0, 2.0, 3.0]),
            "bad": np.array([np.nan, np.nan, np.nan]),
        }

        with pytest.warns(RuntimeWarning):
            result = pipeline.normalize_features(features, method="zscore")

        good = features["good"]
        expected = (good - np.mean(good)) / np.std(good)
        np.testing.assert_allclose(result["good"], expected)
        assert np.all(np.isnan(result["bad"]))

    def test_minmax_nan_ignored(self):
        pipeline = self._pipeline()
        features = {"good": np.array([1.0, 2.0, 3.0, np.nan])}

        result = pipeline.normalize_features(features, method="minmax")

        # NaN is ignored for the min/max bounds: min=1, max=3.
        expected = np.array([0.0, 0.5, 1.0, np.nan])
        np.testing.assert_allclose(result["good"][:3], expected[:3])
        assert np.isnan(result["good"][3])


class TestSyntacticAlignment:
    """Syntactic features aligned to signal samples through the pipeline."""

    def test_aligned_values_at_sample_positions(self, monkeypatch):
        spec = FeatureSpec(
            name="syn",
            extractor_type="syntactic",
            features=["depth"],
            config={"parser_name": "stanford"},
        )
        pipeline = FeaturePipeline(
            PipelineConfig(
                feature_specs=[spec],
                alignment_config={"sampling_rate": 1000.0},
            )
        )
        handler = AlignmentHandler()
        textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)

        # Bypass the external Stanford parser; depth per word position:
        # word 0 -> 3, word 1 -> 2, word 2 -> 1.
        monkeypatch.setattr(
            pipeline._extractors["syn"],
            "extract_to_dict",
            lambda text, features=None: {"depth": {0: 3, 1: 2, 2: 1}},
        )

        result, metadata = pipeline.extract(
            "The cat sat", textgrid=textgrid, signal_length=3000
        )

        assert "syn_depth" in result
        arr = result["syn_depth"]
        assert arr.shape == (3000,)
        assert metadata["aligned"] is True
        assert metadata["n_samples"] == 3000
        np.testing.assert_allclose(arr[0:1000], 3.0)
        np.testing.assert_allclose(arr[1000:2000], 2.0)
        np.testing.assert_allclose(arr[2000:3000], 1.0)


@pytest.mark.llm
class TestLLMValueAlignment:
    """LLM values aligned to signal samples match word-level values."""

    def test_aligned_values_match_word_level(self, tiny_gpt2_path):
        pytest.importorskip("torch")
        from pyeeg.features.llm_features import (
            LLMFeatureConfig,
            LLMFeatureExtractor,
        )

        # Compute the unaligned word-level surprisal values directly.
        llm_config = LLMFeatureConfig(model_name=tiny_gpt2_path, device="cpu")
        extractor = LLMFeatureExtractor(llm_config)
        word_level = extractor.extract("The cat sat", ["surprisal"])["surprisal"]

        spec = FeatureSpec(
            name="llm",
            extractor_type="llm",
            features=["surprisal"],
            config={"model_name": tiny_gpt2_path, "device": "cpu"},
        )
        pipeline = FeaturePipeline(
            PipelineConfig(
                feature_specs=[spec],
                alignment_config={"sampling_rate": 1000.0},
            )
        )
        handler = AlignmentHandler()
        textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)

        result, metadata = pipeline.extract(
            "The cat sat", textgrid=textgrid, signal_length=3000
        )

        assert "llm_surprisal" in result
        arr = result["llm_surprisal"]
        assert arr.shape == (3000,)
        assert metadata["aligned"] is True

        # Word intervals cover [0,1), [1,2), [2,3) seconds at 1000 Hz. The
        # aligned value over each interval must equal the word-level value.
        # NaN word values are skipped when building the alignment map, so the
        # corresponding interval stays at the matrix initialization value 0.0.
        for i in range(len(word_level)):
            slice_ = arr[i * 1000 : (i + 1) * 1000]
            if np.isnan(word_level[i]):
                np.testing.assert_allclose(slice_, 0.0)
            else:
                np.testing.assert_allclose(slice_, float(word_level[i]))


class TestSyntacticAlignmentEdgeCases:
    """Dictionary-form syntactic features with NaN / non-numeric values."""

    def _make_pipeline(self):
        spec = FeatureSpec(
            name="syn",
            extractor_type="syntactic",
            features=["depth"],
            config={"parser_name": "stanford"},
        )
        return FeaturePipeline(
            PipelineConfig(
                feature_specs=[spec],
                alignment_config={"sampling_rate": 1000.0},
            )
        )

    def test_dict_nan_values_are_skipped(self, monkeypatch):
        """NaN values in dictionary-form features are skipped, leaving zeros."""
        pipeline = self._make_pipeline()
        textgrid = AlignmentHandler().load_textgrid_from_string(TEXTGRID_STRING)

        # Word 0 has NaN (skipped → zeros), words 1 and 2 have real values.
        monkeypatch.setattr(
            pipeline._extractors["syn"],
            "extract_to_dict",
            lambda text, features=None: {"depth": {0: np.nan, 1: 2, 2: 1}},
        )

        result, _ = pipeline.extract("The cat sat", textgrid=textgrid, signal_length=3000)
        arr = result["syn_depth"]
        np.testing.assert_allclose(arr[0:1000], 0.0)
        np.testing.assert_allclose(arr[1000:2000], 2.0)
        np.testing.assert_allclose(arr[2000:3000], 1.0)

    def test_dict_non_numeric_values_are_skipped(self, monkeypatch):
        """Non-numeric values in dictionary-form features are skipped."""
        pipeline = self._make_pipeline()
        textgrid = AlignmentHandler().load_textgrid_from_string(TEXTGRID_STRING)

        # Word 1 has a string value (skipped → zeros), words 0 and 2 are fine.
        monkeypatch.setattr(
            pipeline._extractors["syn"],
            "extract_to_dict",
            lambda text, features=None: {"depth": {0: 3, 1: "not_a_number", 2: 1}},
        )

        result, _ = pipeline.extract("The cat sat", textgrid=textgrid, signal_length=3000)
        arr = result["syn_depth"]
        np.testing.assert_allclose(arr[0:1000], 3.0)
        np.testing.assert_allclose(arr[1000:2000], 0.0)
        np.testing.assert_allclose(arr[2000:3000], 1.0)

    def test_dict_invalid_position_keys_are_skipped(self, monkeypatch):
        """Invalid (non-integral) position keys are skipped without crashing."""
        pipeline = self._make_pipeline()
        textgrid = AlignmentHandler().load_textgrid_from_string(TEXTGRID_STRING)

        # Position "bad" is not convertible to int; position 1 is fine.
        monkeypatch.setattr(
            pipeline._extractors["syn"],
            "extract_to_dict",
            lambda text, features=None: {"depth": {"bad": 9, 1: 2, 2: 1}},
        )

        result, _ = pipeline.extract("The cat sat", textgrid=textgrid, signal_length=3000)
        arr = result["syn_depth"]
        assert arr.shape == (3000,)
        np.testing.assert_allclose(arr[1000:2000], 2.0)
        np.testing.assert_allclose(arr[2000:3000], 1.0)