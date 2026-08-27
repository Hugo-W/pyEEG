"""Shared fixtures and configuration for the stimulus feature extraction tests.

The tests use a tiny GPT-2 model (random weights, real BPE tokenizer) saved
locally at ``~/.cache/huggingface/tiny-gpt2-test`` so no HuggingFace Hub
download is needed. Run ``tests/build_test_model.py`` to (re)build it.
"""
import os
from pathlib import Path

import pytest

# Path to the locally-built tiny GPT-2 model (see build_test_model.py).
TINY_GPT2_PATH = os.path.expanduser("~/.cache/huggingface/tiny-gpt2-test")

# Path to a real pre-trained distilgpt2 model (downloaded from ModelScope).
# Present only when the user has downloaded it; tests using it are marked
# ``@pytest.mark.slow`` so they don't run by default.
DISTILGPT2_PATH = os.path.expanduser("~/.cache/huggingface/distilgpt2")

# Short English sample texts used across multiple test modules.
SAMPLE_TEXT = "The cat sat on the mat."
SAMPLE_TEXT_LONG = (
    "The cat sat on the mat. "
    "The dog ran in the park. "
    "She said hello to the boy."
)

# A minimal short-form TextGrid (Praat) string with a "words" tier.
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_gpt2_path():
    """Path to the local tiny GPT-2 model directory.

    If the model is not already present, it is built automatically from
    ``tests/build_test_model.py`` (requires torch/transformers/tokenizers,
    available via ``uv sync --extra features --extra test``). This makes the
    test suite self-sufficient on clean checkouts without network access.
    """
    if not Path(TINY_GPT2_PATH).exists():
        # Auto-build the tiny model if dependencies are available.
        try:
            import importlib
            build_mod = importlib.import_module("build_test_model")
            build_mod.main()
        except Exception as exc:
            pytest.skip(
                f"Could not build tiny-gpt2-test model: {exc}. "
                "Run: uv run --extra features python tests/build_test_model.py"
            )
    return TINY_GPT2_PATH


@pytest.fixture(scope="session")
def llm_extractor(tiny_gpt2_path):
    """A session-scoped LLMFeatureExtractor backed by the tiny local GPT-2."""
    torch = pytest.importorskip("torch")
    from pyeeg.features.llm_features import LLMFeatureConfig, LLMFeatureExtractor

    cfg = LLMFeatureConfig(model_name=tiny_gpt2_path, device="cpu")
    return LLMFeatureExtractor(cfg)


@pytest.fixture
def llm_config_cls():
    """LLMFeatureConfig class (no model loading needed)."""
    pytest.importorskip("torch")
    from pyeeg.features.llm_features import LLMFeatureConfig
    return LLMFeatureConfig


@pytest.fixture
def llm_extractor_cls():
    """LLMFeatureExtractor class."""
    pytest.importorskip("torch")
    from pyeeg.features.llm_features import LLMFeatureExtractor
    return LLMFeatureExtractor


@pytest.fixture(scope="session")
def distilgpt2_path():
    """Path to a real pre-trained distilgpt2 model (if downloaded).

    Skips if the model is not available. To download it, run::

        uv run --extra features python tests/download_distilgpt2.py
    """
    if not Path(DISTILGPT2_PATH).exists():
        pytest.skip(
            "distilgpt2 model not found at "
            f"{DISTILGPT2_PATH}. Run: "
            "uv run --extra features python tests/download_distilgpt2.py"
        )
    return DISTILGPT2_PATH


@pytest.fixture(scope="session")
def distilgpt2_extractor(distilgpt2_path):
    """A session-scoped LLMFeatureExtractor backed by real distilgpt2 weights."""
    torch = pytest.importorskip("torch")
    from pyeeg.features.llm_features import LLMFeatureConfig, LLMFeatureExtractor

    cfg = LLMFeatureConfig(model_name=distilgpt2_path, device="cpu")
    return LLMFeatureExtractor(cfg)
