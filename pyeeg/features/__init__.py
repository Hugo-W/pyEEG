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

from .llm_features import LLMFeatureExtractor
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
