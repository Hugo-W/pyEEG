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

try:
    from .llm_features import LLMFeatureExtractor
except ImportError:
    from warnings import warn
    warn("Missing torch module, will not be able to create LLMFeatureExtractor. Instal torch to use this module (all optional deps installable via natmeeg[features])")
    
    warn("Using a dummy value for LLMFeatureExtractor as torch is unavailable")
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
