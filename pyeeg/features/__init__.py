"""
Stimulus Feature Extraction Module

This module provides tools for extracting features from various types of stimuli:
- Text: LLM-based features (surprisal, entropy, KL divergence)
- Text: Syntactic features (depth, closing nodes, etc.)
- Audio: Alignment with transcripts

Main Classes:
- LLMFeatureExtractor: Extract word-level features using language models
- SyntacticFeatureExtractor: Extract features from constituency trees
- AlignmentHandler: Handle alignment between stimuli and neural data
- FeaturePipeline: Compose multiple feature extractors
"""

from .._logging import LOGGER

try:
    from .llm_features import LLMFeatureExtractor
except ImportError:
    LOGGER.info("Missing torch module, will not be able to create LLMFeatureExtractor. Install torch to use this module (optional deps: pip install natmeeg[features])")

    LOGGER.info("Using a dummy value for LLMFeatureExtractor as torch is unavailable")
    LLMFeatureExtractor = None
from .syntactic_features import SyntacticFeatureExtractor
from .alignment import AlignmentHandler, TextGridParser
from .pipeline import FeaturePipeline
from .reduction import FeatureReducer

__all__ = [
    'LLMFeatureExtractor',
    'SyntacticFeatureExtractor',
    'AlignmentHandler',
    'TextGridParser',
    'FeaturePipeline',
    'FeatureReducer',
]
