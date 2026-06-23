"""
Alignment Handling

This module provides functionality for aligning features with neural signals
using TextGrid files or forced alignment with external tools.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class Interval:
    """Represents an interval in a TextGrid file."""
    start: float
    end: float
    label: str
    tier: str
    
    def duration(self) -> float:
        return self.end - self.start
    
    def contains(self, time: float) -> bool:
        return self.start <= time < self.end


@dataclass
class TextGrid:
    """Represents a TextGrid file with multiple tiers."""
    intervals: Dict[str, List[Interval]]
    start_time: float = 0.0
    end_time: float = 0.0
    
    def get_tier(self, tier_name: str) -> List[Interval]:
        return self.intervals.get(tier_name, [])
    
    def get_word_intervals(self) -> List[Interval]:
        word_tiers = ['words', 'word', 'Word', 'WORDS']
        for tier in word_tiers:
            if tier in self.intervals:
                return self.intervals[tier]
        for tier, intervals in self.intervals.items():
            if intervals and any(' ' in interval.label or interval.label.isalpha() for interval in intervals):
                return intervals
        return []
    
    def get_phone_intervals(self) -> List[Interval]:
        phone_tiers = ['phones', 'phone', 'Phone', 'PHONES', 'phn']
        for tier in phone_tiers:
            if tier in self.intervals:
                return self.intervals[tier]
        return []


class TextGridParser:
    """Parser for Praat TextGrid files."""
    
    def parse_from_string(self, content: str) -> TextGrid:
        """Parse a TextGrid from a string."""
        textgrid = TextGrid()
        textgrid.intervals = {}
        
        lines = content.split('\n')
        current_tier = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            tier_match = re.match(r'item \[(\d+)\]:\s*"(.+?)"', line)
            if tier_match:
                current_tier = tier_match.group(2)
                textgrid.intervals[current_tier] = []
                continue
            
            interval_match = re.match(
                r'intervals \[(\d+)\]:\s*"(.+?)"\s*(\d+\.\d+)\s*(\d+\.\d+)',
                line
            )
            if interval_match and current_tier:
                label = interval_match.group(2)
                start = float(interval_match.group(3))
                end = float(interval_match.group(4))
                textgrid.intervals[current_tier].append(
                    Interval(start=start, end=end, label=label, tier=current_tier)
                )
        
        return textgrid


class AlignmentHandler:
    """Handle alignment between features and neural signals."""
    
    def __init__(self, signal_sampling_rate: float = 1000.0):
        self.sampling_rate = signal_sampling_rate
        self.parser = TextGridParser()
    
    def load_textgrid_from_string(self, content: str) -> TextGrid:
        """Load and parse a TextGrid from a string."""
        return self.parser.parse_from_string(content)
    
    def align_word_features(
        self,
        word_features: Dict[int, Dict[str, float]],
        textgrid: TextGrid,
        signal_length: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Align word-level features to signal time points."""
        word_intervals = textgrid.get_word_intervals()
        
        if not word_intervals:
            logger.error("No word intervals found in TextGrid")
            return np.array([]), []
        
        all_feature_names = set()
        for feat_dict in word_features.values():
            all_feature_names.update(feat_dict.keys())
        all_feature_names = sorted(all_feature_names)
        
        if not all_feature_names:
            logger.warning("No features to align")
            return np.array([]), []
        
        if signal_length is None:
            signal_length = int(textgrid.end_time * self.sampling_rate)
        
        time_points = np.arange(signal_length) / self.sampling_rate
        n_samples = len(time_points)
        n_features = len(all_feature_names)
        
        aligned_features = np.zeros((n_samples, n_features))
        
        for i, interval in enumerate(word_intervals):
            if i not in word_features:
                continue
            
            start_sample = int(interval.start * self.sampling_rate)
            end_sample = int(interval.end * self.sampling_rate)
            
            start_sample = max(0, min(start_sample, n_samples - 1))
            end_sample = max(0, min(end_sample, n_samples))
            
            feat_dict = word_features[i]
            
            for j, feat_name in enumerate(all_feature_names):
                if feat_name in feat_dict:
                    aligned_features[start_sample:end_sample, j] = feat_dict[feat_name]
        
        return aligned_features, all_feature_names