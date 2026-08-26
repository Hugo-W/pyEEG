"""
Alignment Handling

This module provides functionality for aligning features with neural signals
using TextGrid files or forced alignment with external tools.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from .._logging import LOGGER


@dataclass
class Interval:
    """Represents an interval (tier item) in a TextGrid file.

    Parameters
    ----------
    start : float
        Start time of the interval in seconds.
    end : float
        End time of the interval in seconds.
    label : str
        Label of the interval (e.g. the transcribed word or phone).
    tier : str
        Name of the tier this interval belongs to.
    """
    start: float
    end: float
    label: str
    tier: str
    
    def duration(self) -> float:
        """Duration of the interval in seconds.

        Returns
        -------
        duration : float
            Interval length in seconds (``end - start``).
        """
        return self.end - self.start
    
    def contains(self, time: float) -> bool:
        """Check whether a time point falls inside the interval.

        The interval is half-open: ``start <= time < end``.

        Parameters
        ----------
        time : float
            Time point in seconds.

        Returns
        -------
        contains : bool
            ``True`` if ``time`` lies within the interval.
        """
        return self.start <= time < self.end


@dataclass
class TextGrid:
    """Represents a TextGrid file with multiple tiers.

    Parameters
    ----------
    intervals : dict of str -> list of Interval
        Mapping from tier name to the list of intervals in that tier.
    start_time : float
        Start time of the TextGrid in seconds. Default: 0.0.
    end_time : float
        End time of the TextGrid in seconds. Default: 0.0.
    """
    intervals: Dict[str, List[Interval]]
    start_time: float = 0.0
    end_time: float = 0.0
    
    def get_tier(self, tier_name: str) -> List[Interval]:
        """Get the intervals of a named tier.

        Parameters
        ----------
        tier_name : str
            Name of the tier to retrieve.

        Returns
        -------
        intervals : list of Interval
            The intervals in the requested tier, or an empty list if the
            tier is not present.
        """
        return self.intervals.get(tier_name, [])
    
    def get_word_intervals(self) -> List[Interval]:
        """Get the intervals of the word tier.

        Tier names checked first, in order: ``words``, ``word``, ``Word``,
        ``WORDS``. If none of these is present, the first tier whose
        intervals have alphabetic labels or labels containing a space is
        used as a fallback.

        Returns
        -------
        intervals : list of Interval
            Word-level intervals, or an empty list if no word tier is found.
        """
        word_tiers = ['words', 'word', 'Word', 'WORDS']
        for tier in word_tiers:
            if tier in self.intervals:
                return self.intervals[tier]
        for tier, intervals in self.intervals.items():
            if intervals and any(' ' in interval.label or interval.label.isalpha() for interval in intervals):
                return intervals
        return []
    
    def get_phone_intervals(self) -> List[Interval]:
        """Get the intervals of the phone (phoneme) tier.

        Tier names checked, in order: ``phones``, ``phone``, ``Phone``,
        ``PHONES``, ``phn``.

        Returns
        -------
        intervals : list of Interval
            Phone-level intervals, or an empty list if no phone tier is found.
        """
        phone_tiers = ['phones', 'phone', 'Phone', 'PHONES', 'phn']
        for tier in phone_tiers:
            if tier in self.intervals:
                return self.intervals[tier]
        return []


class TextGridParser:
    """Parser for Praat TextGrid files.

    Parses the plain-text (non-binary) TextGrid format, extracting interval
    tiers and their intervals.
    """
    
    def parse_from_string(self, content: str) -> TextGrid:
        """Parse a TextGrid from a string.

        Iterates over the lines of a Praat TextGrid in short text format.
        Tier headers matching ``item [N]: "name"`` start a new tier; lines
        matching ``intervals [N]: "label" start end`` are parsed as
        intervals of the current tier. Tier and interval indices reported by
        Praat are ignored; only names, labels, and timestamps are kept.

        Parameters
        ----------
        content : str
            Full text content of a Praat TextGrid file.

        Returns
        -------
        textgrid : TextGrid
            Parsed TextGrid. ``intervals`` maps tier names to the parsed
            intervals in order of appearance.
        """
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
    """Handle alignment between word-level features and neural signals.

    Converts word-level feature values (indexed by word position) into
    sample-level feature arrays, filling each word's interval with its
    feature values at the configured sampling rate.

    Parameters
    ----------
    signal_sampling_rate : float
        Sampling rate of the neural signal in Hz, used to convert interval
        times (seconds) into sample indices. Default: 1000.0.
    """
    
    def __init__(self, signal_sampling_rate: float = 1000.0):
        self.sampling_rate = signal_sampling_rate
        self.parser = TextGridParser()
    
    def load_textgrid_from_string(self, content: str) -> TextGrid:
        """Load and parse a TextGrid from a string.

        Parameters
        ----------
        content : str
            Full text content of a Praat TextGrid file.

        Returns
        -------
        textgrid : TextGrid
            Parsed TextGrid.
        """
        return self.parser.parse_from_string(content)
    
    def align_word_features(
        self,
        word_features: Dict[int, Dict[str, float]],
        textgrid: TextGrid,
        signal_length: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Align word-level features to signal time points.

        Builds a sample-by-feature matrix where each row corresponds to one
        signal sample and each column to one feature. For every word interval,
        the feature values stored under the interval's index in
        ``word_features`` are written into the rows covering that interval
        (in seconds) converted to sample indices at ``self.sampling_rate``.
        Words without an entry in ``word_features`` are skipped and left as
        zeros. Feature columns are sorted alphabetically.

        Parameters
        ----------
        word_features : dict of int -> dict of str -> float
            Mapping from word index (position of the word interval in the
            word tier) to a dict of feature name -> value for that word.
        textgrid : TextGrid
            TextGrid providing the word intervals, retrieved via
            :meth:`TextGrid.get_word_intervals`.
        signal_length : int, optional
            Number of samples in the output. If ``None``, derived from the
            TextGrid end time and the sampling rate:
            ``int(textgrid.end_time * self.sampling_rate)``.

        Returns
        -------
        aligned_features : ndarray, shape (n_samples, n_features)
            Sample-level feature matrix. Rows are time points
            ``arange(signal_length) / self.sampling_rate``; columns are the
            sorted feature names. Empty (shape ``(0,)``) if no word intervals
            or no features were found.
        feature_names : list of str
            Alphabetically sorted names of the aligned features, matching the
            columns of ``aligned_features``. Empty if no features were found.
        """
        word_intervals = textgrid.get_word_intervals()
        
        if not word_intervals:
            LOGGER.error("No word intervals found in TextGrid")
            return np.array([]), []
        
        all_feature_names = set()
        for feat_dict in word_features.values():
            all_feature_names.update(feat_dict.keys())
        all_feature_names = sorted(all_feature_names)
        
        if not all_feature_names:
            LOGGER.warning("No features to align")
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