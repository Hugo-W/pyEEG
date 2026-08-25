"""
Feature Extraction Pipeline

This module provides a pipeline for composing multiple feature extractors
and aligning their outputs with neural signals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .._logging import LOGGER

try:
    from .llm_features import LLMFeatureExtractor, LLMFeatureConfig
    LLMEXTRACTOR_AVAILABLE = True
except ImportError:
    LLMEXTRACTOR_AVAILABLE = False
from .syntactic_features import SyntacticFeatureExtractor, ParserConfig
from .alignment import AlignmentHandler, TextGrid


@dataclass
class FeatureSpec:
    """Specification for a feature to be extracted."""
    name: str
    extractor_type: str
    features: List[str] = field(default_factory=list)
    config: Optional[Dict] = None


@dataclass
class PipelineConfig:
    """Configuration for the feature extraction pipeline."""
    feature_specs: List[FeatureSpec] = field(default_factory=list)
    alignment_config: Optional[Dict] = None
    normalization: str = "none"
    cache_features: bool = True


class FeaturePipeline:
    """Pipeline for extracting and aligning multiple features from stimuli."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._extractors: Dict[str, object] = {}
        self._alignment_handler: Optional[AlignmentHandler] = None
        self._initialize_extractors()
        self._initialize_alignment()
    
    def _initialize_extractors(self):
        """Initialize all feature extractors."""
        for spec in self.config.feature_specs:
            if spec.extractor_type == 'llm':
                llm_config = LLMFeatureConfig(
                    model_name=spec.config.get('model_name', 'GroNLP/gpt2-small-dutch'),
                    device=spec.config.get('device', 'cpu')
                )
                if LLMEXTRACTOR_AVAILABLE:
                    self._extractors[spec.name] = LLMFeatureExtractor(llm_config)
            elif spec.extractor_type == 'syntactic':
                parser_config = ParserConfig(
                    parser_name=spec.config.get('parser_name', 'stanford'),
                    language=spec.config.get('language', 'en')
                )
                self._extractors[spec.name] = SyntacticFeatureExtractor(parser_config)
    
    def _initialize_alignment(self):
        """Initialize alignment handler."""
        if self.config.alignment_config:
            sampling_rate = self.config.alignment_config.get('sampling_rate', 1000.0)
            self._alignment_handler = AlignmentHandler(sampling_rate)
    
    def extract(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract features from text and optionally align to signal."""
        metadata = {
            'text': text,
            'text_length': len(text),
            'feature_specs': [spec.name for spec in self.config.feature_specs],
            'aligned': textgrid is not None
        }
        
        all_features: Dict[str, Dict] = {}
        
        for spec in self.config.feature_specs:
            extractor = self._extractors.get(spec.name)
            if extractor is None:
                continue
            
            if spec.extractor_type == 'llm':
                features = extractor.extract(text, spec.features)
            elif spec.extractor_type == 'syntactic':
                features = extractor.extract_to_dict(text, spec.features)
            else:
                continue
            
            if features:
                all_features[spec.name] = features
        
        if textgrid is None and self._alignment_handler is None:
            result = {}
            for extractor_name, feat_dict in all_features.items():
                for feat_name, values in feat_dict.items():
                    if isinstance(values, dict):
                        words = text.split()
                        arr = np.zeros(len(words))
                        for pos, val in values.items():
                            if pos < len(words):
                                arr[pos] = val
                        result[f"{extractor_name}_{feat_name}"] = arr
                    elif isinstance(values, np.ndarray):
                        result[f"{extractor_name}_{feat_name}"] = values
            metadata['aligned'] = False
            return result, metadata
        
        if textgrid is None:
            LOGGER.warning("No TextGrid provided and no alignment handler configured")
            textgrid = TextGrid()
        
        result = {}
        for extractor_name, feat_dict in all_features.items():
            aligned, feat_names = self._alignment_handler.align_word_features(
                feat_dict, textgrid, signal_length
            )
            for i, feat_name in enumerate(feat_names):
                result[f"{extractor_name}_{feat_name}"] = aligned[:, i]
        
        metadata['aligned'] = True
        metadata['n_samples'] = aligned.shape[0] if aligned.size > 0 else 0
        metadata['sampling_rate'] = self._alignment_handler.sampling_rate
        
        return result, metadata
    
    def normalize_features(
        self,
        features: Dict[str, np.ndarray],
        method: str = "zscore"
    ) -> Dict[str, np.ndarray]:
        """Normalize features."""
        if method == 'none':
            return features
        
        result = {}
        for name, arr in features.items():
            if method == 'zscore':
                mean = np.mean(arr)
                std = np.std(arr)
                if std > 0:
                    result[name] = (arr - mean) / std
                else:
                    result[name] = arr - mean
            elif method == 'minmax':
                min_val = np.min(arr)
                max_val = np.max(arr)
                if max_val > min_val:
                    result[name] = (arr - min_val) / (max_val - min_val)
                else:
                    result[name] = arr - min_val
            else:
                result[name] = arr
        return result


class StimulusEncoder:
    """High-level interface for stimulus feature extraction."""
    
    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        if pipeline_config is None:
            pipeline_config = PipelineConfig()
        self.pipeline = FeaturePipeline(pipeline_config)
    
    def add_llm_features(
        self,
        features: List[str] = None,
        model_name: str = "GroNLP/gpt2-small-dutch",
        name: str = "llm"
    ):
        """Add LLM-based features to the pipeline."""
        if features is None:
            features = ['surprisal', 'entropy', 'kl_divergence']
        spec = FeatureSpec(
            name=name,
            extractor_type='llm',
            features=features,
            config={'model_name': model_name}
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()
    
    def add_syntactic_features(
        self,
        features: List[str] = None,
        parser_name: str = "stanford",
        name: str = "syntactic"
    ):
        """Add syntactic features to the pipeline."""
        if features is None:
            features = ['depth', 'opening', 'closing']
        spec = FeatureSpec(
            name=name,
            extractor_type='syntactic',
            features=features,
            config={'parser_name': parser_name}
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()
    
    def set_alignment(self, sampling_rate: float = 1000.0):
        """Set alignment configuration."""
        self.pipeline.config.alignment_config = {
            'sampling_rate': sampling_rate
        }
        self.pipeline._initialize_alignment()
    
    def encode(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract and align features from text."""
        features, metadata = self.pipeline.extract(
            text, textgrid, signal_length
        )
        if self.pipeline.config.normalization != 'none':
            features = self.pipeline.normalize_features(
                features,
                self.pipeline.config.normalization
            )
        return features, metadata
    
    def encode_to_array(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> np.ndarray:
        """Extract features and return as a single array."""
        features, metadata = self.encode(text, textgrid, signal_length)
        if not features:
            return np.array([])
        feature_names = sorted(features.keys())
        stacked = np.column_stack([features[name] for name in feature_names])
        return stacked