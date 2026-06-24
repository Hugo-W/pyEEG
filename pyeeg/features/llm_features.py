"""
LLM-Based Feature Extraction

This module provides functionality for extracting word-level linguistic features
using language models, incorporating code from NNLM/lm_featurize.

Features:
- Surprisal: Information content of each word given its context (-log2(P(word | context)))
- Entropy: Uncertainty in the token distribution (-sum(p * log2(p)))
- KL Divergence: Difference between token distributions (KL(P||Q))
- Prediction Error: Surprisal normalized by entropy

Based on NNLM/lm_featurize by Hugo Weissbart
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
try:
    import torch
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("torch is not installed. Instal torch to use this module (all optional deps installable via natmeeg[features])")

logger = logging.getLogger(__name__)


# Special tokens used in BPE tokenization
SEPARATOR = '\u0160'  # 'Ġ' - Byte pair encoding separator
NEWLINE = '\u010a'   # 'Ċ' - Newline token


@dataclass
class LLMFeatureConfig:
    """Configuration for LLM feature extraction."""
    model_name: str = "GroNLP/gpt2-small-dutch"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Optional[str] = None
    use_cache: bool = True
    return_shape: str = "valid"


class LLMFeatureExtractor:
    """
    Extract word-level linguistic features using language models.
    
    Based on: NNLM/lm_featurize/metrics.py by Hugo Weissbart
    
    Args:
        config: Configuration for the feature extractor
    """
    
    def __init__(self, config: LLMFeatureConfig = None):
        if config is None:
            config = LLMFeatureConfig()
        self.config = config
        self._tokenizer = None
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModel
        except ImportError:
            logger.error("transformers library not installed")
            raise
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if "gpt2" in self.config.model_name.lower():
            self._model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        else:
            self._model = AutoModel.from_pretrained(self.config.model_name)
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        
        logger.info(f"Loaded model: {self.config.model_name}")
    
    def get_surprisal(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                    return_shape: str = None) -> torch.Tensor:
        """
        Calculate surprisal (-log(p)) for tokens.
        Based on: NNLM/lm_featurize/metrics.py
        """
        if return_shape is None:
            return_shape = self.config.return_shape
        
        p = F.softmax(logits, dim=1)
        
        if return_shape == 'valid':
            return -p[:-1, input_ids.flatten()[1:]].diag().log()
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, -p[:-1, input_ids.flatten()[1:]].diag().log()))
        elif return_shape == 'full':
            return -p[:, input_ids.flatten()].diag().log()
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")
    
    def get_entropy(self, logits: torch.Tensor, return_shape: str = None) -> torch.Tensor:
        """Calculate entropy of token distributions. Based on: NNLM/lm_featurize/metrics.py"""
        if return_shape is None:
            return_shape = self.config.return_shape
        
        if return_shape == 'valid':
            return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)[:-1]
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)[:-1]))
        elif return_shape == 'full':
            return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")
    
    def get_kl_divergence(self, logits: torch.Tensor, return_shape: str = None) -> torch.Tensor:
        """Compute KL divergence between current and previous state. Based on: NNLM/lm_featurize/metrics.py"""
        if return_shape is None:
            return_shape = self.config.return_shape
        
        P = F.log_softmax(logits, dim=1)
        
        if return_shape == 'valid':
            return F.kl_div(P[:1, :], P[1:, :], reduction='none', log_target=True).sum(1)
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, F.kl_div(P[:-1, :], P[1:, :], reduction='none', log_target=True).sum(1)))
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")
    
    def get_prediction_error(self, surprisal: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """Calculate prediction error (surprisal/entropy)."""
        entropy_safe = entropy.clone()
        entropy_safe[entropy_safe == 0] = 1e-10
        return surprisal / entropy_safe
    
    def get_tokens(self, input_ids: torch.Tensor) -> List[str]:
        """Convert input IDs to token strings."""
        return self._tokenizer.convert_ids_to_tokens(input_ids.flatten())
    
    def bpe_to_words(self, tokens: List[str], sep: str = SEPARATOR, newline: str = NEWLINE, 
                     lang: str = 'nl') -> List[List[int]]:
        """
        Map BPE tokens to word indices. Based on: NNLM/lm_featurize/utils.py
        """
        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            logger.error("NLTK not installed")
            raise
        
        tokens_clean = word_tokenize(''.join(tokens).replace(sep, ' ').replace(newline, '\n'), language=lang)
        
        indices_map = []
        k = 0
        
        for t in tokens_clean:
            bytepairs = []
            test_token = tokens[k].strip(sep).strip(newline)
            if test_token != '':
                bytepairs.append(k)
            
            while test_token != t:
                k += 1
                if k == len(tokens): break
                test_token += tokens[k].strip(sep).strip(newline)
                if test_token != '':
                    bytepairs.append(k)
                if len(bytepairs) > 100:
                    raise RecursionError(f"Can't match token: {t}")
            
            indices_map.append(bytepairs)
            k += 1
            if k >= len(tokens): break
        
        return indices_map
    
    def reduce_features(self, features: Dict[str, np.ndarray], indices_map: List[List[int]]) -> Dict[str, np.ndarray]:
        """Reduce BPE-level features to word-level. Based on: NNLM/lm_featurize/utils.py"""
        from functools import reduce as functools_reduce
        
        def sum_reduce(x):
            return functools_reduce(lambda x, y: x+y, x)
        
        reduced_features = {}
        
        for feat_name, feat_array in features.items():
            if feat_name == 'tokens':
                reduced_tokens = []
                for bpe_indices in indices_map:
                    token_str = ''.join([features['tokens'][i] for i in bpe_indices]).replace(sep, '').replace(newline, '')
                    reduced_tokens.append(token_str)
                reduced_features[feat_name] = np.array(reduced_tokens)
            elif feat_name == 'entropy':
                reduced_values = []
                for bpe_indices in indices_map:
                    reduced_values.append(feat_array[bpe_indices[-1]])
                reduced_features[feat_name] = np.array(reduced_values)
            elif feat_name == 'kl_divergence':
                reduced_values = []
                for bpe_indices in indices_map:
                    if len(bpe_indices) > 1:
                        reduced_values.append(feat_array[bpe_indices[0]] + feat_array[bpe_indices[-1]])
                    else:
                        reduced_values.append(feat_array[bpe_indices[0]])
                reduced_features[feat_name] = np.array(reduced_values)
            else:
                reduced_values = []
                for bpe_indices in indices_map:
                    reduced_values.append(sum_reduce(feat_array[bpe_indices]))
                reduced_features[feat_name] = np.array(reduced_values)
        
        if 'surprisal' in reduced_features and 'entropy' in reduced_features:
            entropy_safe = reduced_features['entropy'].copy()
            entropy_safe[entropy_safe == 0] = 1e-10
            reduced_features['prediction_error'] = reduced_features['surprisal'] / entropy_safe
        
        return reduced_features
    
    def extract(self, text: str, features: List[str] = None, 
                return_word_level: bool = True) -> Dict[str, np.ndarray]:
        """Extract features from text."""
        if features is None:
            features = ['surprisal', 'entropy', 'kl_divergence', 'prediction_error']
        
        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids']
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
        
        result = {}
        
        if 'surprisal' in features or 'prediction_error' in features:
            result['surprisal'] = self.get_surprisal(logits, input_ids, 'same').detach().cpu().numpy()
        
        if 'entropy' in features or 'prediction_error' in features:
            result['entropy'] = self.get_entropy(logits, 'same').detach().cpu().numpy()
        
        if 'kl_divergence' in features:
            result['kl_divergence'] = self.get_kl_divergence(logits, 'same').detach().cpu().numpy()
        
        if 'tokens' in features:
            result['tokens'] = np.array(self.get_tokens(input_ids))
        
        if return_word_level and len(result) > 0:
            tokens = self.get_tokens(input_ids)
            indices_map = self.bpe_to_words(tokens, lang='en')
            result = self.reduce_features(result, indices_map)
        
        return result