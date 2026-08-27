# -*- coding: utf-8 -*-
"""Unit tests for pyeeg.features.llm_features.

Two tiers of tests:

1. Synthetic tensor tests (no model needed): exercise the metric methods
   (get_surprisal, get_entropy, get_kl_divergence, get_prediction_error) and
   reduce_features with hand-crafted tensors/arrays.
2. Local model tests (marked ``@pytest.mark.llm``): exercise ``extract()``
   end-to-end using the tiny local GPT-2 model provided by the ``llm_extractor``
   fixture.
"""
import numpy as np
import pytest

# The llm_features module imports torch at import time, so skip the whole test
# module when torch is unavailable.
torch = pytest.importorskip("torch")

from numpy.testing import assert_array_equal  # noqa: E402

from conftest import SAMPLE_TEXT  # noqa: E402
from pyeeg.features.llm_features import (  # noqa: E402
    LLMFeatureConfig,
    LLMFeatureExtractor,
    SEPARATOR,
)

# Shape helpers for the synthetic metric tests: seq_len=4, vocab_size=3.
LOGITS = torch.tensor(
    [
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.5, 1.5, 2.5],
    ]
)
INPUT_IDS = torch.tensor([[0, 1, 2, 0]])


@pytest.fixture
def bare_extractor(llm_extractor_cls, llm_config_cls):
    """An LLMFeatureExtractor instance without a loaded model.

    The metric methods and reduce_features only need ``self.config``, so we
    bypass ``__init__`` (which would load a language model) via ``__new__``.
    """
    ext = llm_extractor_cls.__new__(llm_extractor_cls)
    ext.config = llm_config_cls()
    return ext


# ===========================================================================
# Synthetic tensor tests (no model needed)
# ===========================================================================


class TestGetSurprisal:
    """Tests for get_surprisal with hand-crafted logits."""

    def test_valid_shape(self, bare_extractor):
        out = bare_extractor.get_surprisal(LOGITS, INPUT_IDS, "valid")
        assert out.shape == (3,)
        assert torch.isnan(out).sum() == 0

    def test_same_shape(self, bare_extractor):
        out = bare_extractor.get_surprisal(LOGITS, INPUT_IDS, "same")
        assert out.shape == (4,)
        assert torch.isnan(out[0])
        assert not torch.isnan(out[1:]).any()

    def test_full_shape(self, bare_extractor):
        out = bare_extractor.get_surprisal(LOGITS, INPUT_IDS, "full")
        assert out.shape == (4,)
        assert not torch.isnan(out).any()

    def test_flat_input_ids(self, bare_extractor):
        out = bare_extractor.get_surprisal(LOGITS, INPUT_IDS.flatten(), "valid")
        assert out.shape == (3,)

    def test_unknown_return_shape_raises(self, bare_extractor):
        with pytest.raises(ValueError):
            bare_extractor.get_surprisal(LOGITS, INPUT_IDS, "bogus")

    def test_nonnegative(self, bare_extractor):
        out = bare_extractor.get_surprisal(LOGITS, INPUT_IDS, "valid")
        assert torch.all(out >= 0)


class TestGetEntropy:
    """Tests for get_entropy with hand-crafted logits."""

    def test_valid_shape(self, bare_extractor):
        out = bare_extractor.get_entropy(LOGITS, "valid")
        assert out.shape == (3,)
        assert torch.isnan(out).sum() == 0

    def test_same_shape(self, bare_extractor):
        out = bare_extractor.get_entropy(LOGITS, "same")
        assert out.shape == (4,)
        assert torch.isnan(out[0])
        assert not torch.isnan(out[1:]).any()

    def test_full_shape(self, bare_extractor):
        out = bare_extractor.get_entropy(LOGITS, "full")
        assert out.shape == (4,)
        assert not torch.isnan(out).any()

    def test_unknown_return_shape_raises(self, bare_extractor):
        with pytest.raises(ValueError):
            bare_extractor.get_entropy(LOGITS, "bogus")

    def test_nonnegative(self, bare_extractor):
        out = bare_extractor.get_entropy(LOGITS, "valid")
        assert torch.all(out >= 0)

    def test_uniform_logits_give_log_vocab(self, bare_extractor):
        uniform = torch.zeros(4, 3)
        out = bare_extractor.get_entropy(uniform, "valid")
        # Uniform distribution over 3 tokens -> entropy = log(3).
        assert torch.allclose(out, torch.full((3,), float(np.log(3))))


class TestGetKLDivergence:
    """Tests for get_kl_divergence with hand-crafted logits."""

    def test_valid_shape(self, bare_extractor):
        out = bare_extractor.get_kl_divergence(LOGITS, "valid")
        assert out.shape == (3,)
        assert torch.isnan(out).sum() == 0

    def test_same_shape(self, bare_extractor):
        out = bare_extractor.get_kl_divergence(LOGITS, "same")
        assert out.shape == (4,)
        assert torch.isnan(out[0])
        assert not torch.isnan(out[1:]).any()

    def test_full_raises(self, bare_extractor):
        with pytest.raises(ValueError):
            bare_extractor.get_kl_divergence(LOGITS, "full")

    def test_unknown_return_shape_raises(self, bare_extractor):
        with pytest.raises(ValueError):
            bare_extractor.get_kl_divergence(LOGITS, "bogus")

    def test_nonnegative(self, bare_extractor):
        out = bare_extractor.get_kl_divergence(LOGITS, "valid")
        assert torch.all(out >= 0)


class TestGetPredictionError:
    """Tests for get_prediction_error."""

    def test_basic_ratio(self, bare_extractor):
        surprisal = torch.tensor([1.0, 2.0, 3.0])
        entropy = torch.tensor([0.5, 2.0, 1.0])
        out = bare_extractor.get_prediction_error(surprisal, entropy)
        assert torch.allclose(out, torch.tensor([2.0, 1.0, 3.0]))

    def test_zero_entropy_does_not_divide_by_zero(self, bare_extractor):
        surprisal = torch.tensor([1.0, 2.0, 3.0])
        entropy = torch.tensor([0.0, 0.0, 1.0])
        out = bare_extractor.get_prediction_error(surprisal, entropy)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        # Zero entropy is replaced with 1e-10 -> large but finite ratio.
        assert torch.allclose(out[0], torch.tensor(1.0 / 1e-10))
        assert torch.allclose(out[1], torch.tensor(2.0 / 1e-10))
        assert torch.allclose(out[2], torch.tensor(3.0))

    def test_no_mutation_of_input(self, bare_extractor):
        entropy = torch.tensor([0.0, 1.0])
        surprisal = torch.tensor([1.0, 1.0])
        bare_extractor.get_prediction_error(surprisal, entropy)
        assert torch.equal(entropy, torch.tensor([0.0, 1.0]))


class TestReduceFeatures:
    """Tests for reduce_features (BPE -> word aggregation)."""

    def make_features(self):
        return {
            "tokens": np.array(["The", f"{SEPARATOR}cat", f"{SEPARATOR}sat"]),
            "surprisal": np.array([1.0, 2.0, 3.0]),
            "entropy": np.array([0.5, 1.0, 1.5]),
        }

    def test_single_token_words(self, bare_extractor):
        features = self.make_features()
        indices_map = [[0], [1], [2]]
        reduced = bare_extractor.reduce_features(features, indices_map)

        # Surprisal is summed -> unchanged for single-token words.
        assert_array_equal(reduced["surprisal"], np.array([1.0, 2.0, 3.0]))
        # Entropy takes the last token's value -> unchanged here.
        assert_array_equal(reduced["entropy"], np.array([0.5, 1.0, 1.5]))
        # tokens strip the BPE separator marker.
        assert_array_equal(reduced["tokens"], np.array(["The", "cat", "sat"]))

    def test_prediction_error_added_when_both_present(self, bare_extractor):
        features = self.make_features()
        indices_map = [[0], [1], [2]]
        reduced = bare_extractor.reduce_features(features, indices_map)
        assert "prediction_error" in reduced
        assert_array_equal(
            reduced["prediction_error"],
            np.array([1.0 / 0.5, 2.0 / 1.0, 3.0 / 1.5]),
        )

    def test_no_prediction_error_without_entropy(self, bare_extractor):
        features = {
            "tokens": np.array(["The", f"{SEPARATOR}cat"]),
            "surprisal": np.array([1.0, 2.0]),
        }
        reduced = bare_extractor.reduce_features(features, [[0], [1]])
        assert "prediction_error" not in reduced

    def test_multi_token_words(self, bare_extractor):
        features = self.make_features()
        indices_map = [[0, 1], [2]]
        reduced = bare_extractor.reduce_features(features, indices_map)

        # Surprisal sums the word's tokens: 1.0 + 2.0 = 3.0, then 3.0.
        assert_array_equal(reduced["surprisal"], np.array([3.0, 3.0]))
        # Entropy takes the last token of each word: 1.0, then 1.5.
        assert_array_equal(reduced["entropy"], np.array([1.0, 1.5]))
        # tokens concatenate subwords and strip the separator marker.
        assert_array_equal(reduced["tokens"], np.array(["Thecat", "sat"]))


# ===========================================================================
# Local model tests (end-to-end with the tiny local GPT-2)
# ===========================================================================


@pytest.mark.llm
class TestExtract:
    """End-to-end tests for extract() using the tiny local GPT-2."""

    def test_word_level_features(self, llm_extractor):
        res = llm_extractor.extract(SAMPLE_TEXT)
        assert set(res.keys()) == {
            "surprisal",
            "entropy",
            "kl_divergence",
            "prediction_error",
        }
        for name, values in res.items():
            assert isinstance(values, np.ndarray)
            assert values.ndim == 1
        # Leading NaN for the first word (no preceding context).
        assert np.isnan(res["surprisal"][0])
        assert np.isnan(res["kl_divergence"][0])
        assert np.isnan(res["prediction_error"][0])
        # "The cat sat on the mat." -> 6 words ("." merges with "mat").
        assert len(res["surprisal"]) == 6
        # Surprisal is non-negative (ignoring the leading NaN).
        assert np.all(np.nan_to_num(res["surprisal"], nan=0.0) >= 0)
        # Entropy is non-negative too.
        assert np.all(np.nan_to_num(res["entropy"], nan=0.0) >= 0)

    def test_token_level_features(self, llm_extractor):
        word_level = llm_extractor.extract(SAMPLE_TEXT)
        res = llm_extractor.extract(SAMPLE_TEXT, return_word_level=False)
        # More BPE tokens than words.
        assert len(res["surprisal"]) > len(word_level["surprisal"])
        # Leading NaN for the first BPE token.
        assert np.isnan(res["surprisal"][0])

    def test_specific_features(self, llm_extractor):
        res = llm_extractor.extract(SAMPLE_TEXT, features=["surprisal"])
        assert set(res.keys()) == {"surprisal"}
        res = llm_extractor.extract(SAMPLE_TEXT, features=["entropy"])
        assert set(res.keys()) == {"entropy"}
        res = llm_extractor.extract(SAMPLE_TEXT, features=["tokens"])
        assert set(res.keys()) == {"tokens"}
        assert res["tokens"].dtype.kind in ("U", "S", "O")
        assert all(isinstance(t, str) for t in res["tokens"].tolist())

    def test_empty_features(self, llm_extractor):
        assert llm_extractor.extract(SAMPLE_TEXT, features=[]) == {}


@pytest.mark.llm
class TestTokenizationHelpers:
    """Tests for the tokenizer-facing helpers."""

    def test_bpe_to_words_fast(self, llm_extractor):
        tokens = llm_extractor.get_tokens(
            llm_extractor._tokenizer(SAMPLE_TEXT, return_tensors="pt")["input_ids"]
        )
        indices_map = llm_extractor.bpe_to_words_fast(SAMPLE_TEXT, tokens)
        assert isinstance(indices_map, list)
        for word in indices_map:
            assert isinstance(word, list)
            for idx in word:
                assert isinstance(idx, int)
        # Every BPE token belongs to exactly one word.
        assert sum(len(word) for word in indices_map) == len(tokens)

    def test_get_tokens(self, llm_extractor):
        input_ids = llm_extractor._tokenizer(SAMPLE_TEXT, return_tensors="pt")[
            "input_ids"
        ]
        tokens = llm_extractor.get_tokens(input_ids)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert len(tokens) == input_ids.numel()

    def test_tokenizer_is_fast(self, llm_extractor):
        assert llm_extractor._tokenizer.is_fast is True


# ===========================================================================
# Additional end-to-end tests (appended)
# ===========================================================================

from conftest import SAMPLE_TEXT_LONG  # noqa: E402


@pytest.mark.llm
class TestExtractPunctuation:
    """Punctuation handling in word-level extraction."""

    def test_punctuation_handling(self, llm_extractor):
        res = llm_extractor.extract("She said: hello!")
        # The BPE tokenizer may merge punctuation with the preceding word;
        # we only require a reasonable number of words and no crash.
        assert len(res["surprisal"]) >= 3
        # All numeric features must be finite (ignoring leading NaN).
        for name, values in res.items():
            assert isinstance(values, np.ndarray)
            assert values.ndim == 1
            assert np.all(np.isfinite(values[~np.isnan(values)]))


@pytest.mark.llm
class TestExtractMultiSentence:
    """Multi-sentence extraction produces more word-level features."""

    def test_multi_sentence_more_words_than_single(self, llm_extractor):
        single = llm_extractor.extract(SAMPLE_TEXT)
        multi = llm_extractor.extract(SAMPLE_TEXT_LONG)
        assert len(multi["surprisal"]) > len(single["surprisal"])
        # All values finite for the longer text (ignoring leading NaN).
        for name, values in multi.items():
            assert np.all(np.isfinite(values[~np.isnan(values)]))


@pytest.mark.llm
class TestTokenVsWordCount:
    """Token-level output has more entries than word-level output."""

    def test_token_count_exceeds_word_count(self, llm_extractor):
        token_result = llm_extractor.extract(SAMPLE_TEXT, return_word_level=False)
        word_result = llm_extractor.extract(SAMPLE_TEXT)
        assert len(token_result["surprisal"]) > len(word_result["surprisal"])


@pytest.mark.llm
class TestSurprisalNonNegative:
    """Surprisal is always non-negative for every word."""

    def test_word_level_surprisal_nonnegative(self, llm_extractor):
        res = llm_extractor.extract(SAMPLE_TEXT, features=["surprisal"])
        surprisal = res["surprisal"]
        valid = surprisal[~np.isnan(surprisal)]
        assert np.all(valid >= 0)


@pytest.mark.llm
class TestPredictionErrorOnly:
    """Requesting only prediction_error still yields the required internals."""

    def test_prediction_error_requires_surprisal_and_entropy(self, llm_extractor):
        res = llm_extractor.extract(
            SAMPLE_TEXT, features=["prediction_error"]
        )
        # prediction_error requires surprisal and entropy internally.
        assert set(res.keys()) == {"surprisal", "entropy", "prediction_error"}
        surprisal = res["surprisal"]
        entropy = res["entropy"]
        pred_error = res["prediction_error"]
        # prediction_error = surprisal / entropy_safe for each word, where
        # zero entropy is replaced with 1e-10 (mirrors reduce_features).
        mask = ~np.isnan(surprisal) & ~np.isnan(entropy) & ~np.isnan(pred_error)
        entropy_safe = np.where(entropy == 0, 1e-10, entropy)
        assert np.allclose(pred_error[mask], surprisal[mask] / entropy_safe[mask])


# ---------------------------------------------------------------------------
# Real-model tests (distilgpt2) — marked slow, require downloaded weights.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDistilGPT2:
    """End-to-end tests with real pre-trained distilgpt2 weights.

    These validate that the extraction pipeline produces linguistically
    meaningful features (not just shape/type checks).  The model (~334 MB) is
    downloaded from ModelScope; run ``tests/download_distilgpt2.py`` first.

    Skipped if the model is not available.
    """

    def test_common_word_has_lower_surprisal(self, distilgpt2_extractor):
        """Common words like "the" should have lower surprisal than rare words."""
        result = distilgpt2_extractor.extract(
            "The cat sat on the mat.", return_word_level=True
        )
        surp = result["surprisal"]
        valid = surp[~np.isnan(surp)]
        assert len(valid) >= 5  # at least 5 words with surprisal
        # "the" (index 4, 0-indexed) should have low surprisal — it's very
        # predictable after "sat on".  "mat" (index 5) should be higher.
        assert surp[4] < surp[5], (
            f"'the' surprisal ({surp[4]}) should be < 'mat' surprisal ({surp[5]})"
        )

    def test_surprisal_is_finite_and_nonneg(self, distilgpt2_extractor):
        """All non-NaN surprisal values are finite and non-negative."""
        result = distilgpt2_extractor.extract(
            "The cat sat on the mat.", return_word_level=True
        )
        surp = result["surprisal"]
        valid = surp[~np.isnan(surp)]
        assert np.all(np.isfinite(valid))
        assert np.all(valid >= 0)

    def test_surprisal_varies_across_words(self, distilgpt2_extractor):
        """Real model should produce varied surprisal (std > 0)."""
        result = distilgpt2_extractor.extract(
            "The cat sat on the mat. The dog ran in the park.",
            return_word_level=True,
        )
        surp = result["surprisal"]
        valid = surp[~np.isnan(surp)]
        assert valid.std() > 0.5, (
            f"Surprisal std={valid.std():.3f} is too low — model may not be working"
        )

    def test_token_level_has_more_entries_than_word_level(self, distilgpt2_extractor):
        """BPE token count exceeds word count."""
        text = "The cat sat on the mat."
        word = distilgpt2_extractor.extract(text, return_word_level=True)
        token = distilgpt2_extractor.extract(text, return_word_level=False)
        assert len(token["surprisal"]) >= len(word["surprisal"])

    def test_entropy_is_positive(self, distilgpt2_extractor):
        """Entropy of the predictive distribution should be positive."""
        result = distilgpt2_extractor.extract(
            "The cat sat on the mat.", return_word_level=True
        )
        ent = result["entropy"]
        valid = ent[~np.isnan(ent)]
        assert np.all(valid > 0)

    def test_pipeline_with_distilgpt2(self, distilgpt2_path):
        """Full pipeline end-to-end with real model weights."""
        pytest.importorskip("torch")
        from pyeeg.features.pipeline import (
            FeaturePipeline, FeatureSpec, PipelineConfig, StimulusEncoder,
        )

        encoder = StimulusEncoder()
        encoder.add_llm_features(
            features=["surprisal", "entropy"], model_name=distilgpt2_path
        )
        features, meta = encoder.encode("The cat sat on the mat.")
        assert "llm_surprisal" in features
        assert "llm_entropy" in features
        assert features["llm_surprisal"].ndim == 1