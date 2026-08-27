"""Unit tests for the alignment module (``pyeeg.features.alignment``).

Covers the ``Interval`` and ``TextGrid`` dataclasses, the ``TextGridParser``
(long-format Praat TextGrid parsing) and the ``AlignmentHandler`` that maps
word-level features onto sample-level arrays. These tests are pure Python and
do not require torch or any language model.
"""

import numpy as np
import pytest

from conftest import TEXTGRID_STRING
from pyeeg.features.alignment import (
    AlignmentHandler,
    Interval,
    TextGrid,
    TextGridParser,
)


# ---------------------------------------------------------------------------
# Interval dataclass
# ---------------------------------------------------------------------------

def test_interval_duration():
    interval = Interval(start=1.0, end=2.5, label="cat", tier="words")
    assert interval.duration() == pytest.approx(1.5)


def test_interval_contains_is_half_open():
    interval = Interval(start=1.0, end=2.0, label="cat", tier="words")
    assert interval.contains(1.0)
    assert interval.contains(1.5)
    assert not interval.contains(2.0)  # end is exclusive
    assert not interval.contains(0.5)
    assert not interval.contains(2.5)


# ---------------------------------------------------------------------------
# TextGridParser
# ---------------------------------------------------------------------------

def test_parse_from_string_returns_textgrid():
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    assert isinstance(textgrid, TextGrid)


def test_parse_words_tier():
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    intervals = textgrid.get_tier("words")
    assert len(intervals) == 3
    starts = [interval.start for interval in intervals]
    ends = [interval.end for interval in intervals]
    labels = [interval.label for interval in intervals]
    assert starts == [0.0, 1.0, 2.0]
    assert ends == [1.0, 2.0, 3.0]
    assert labels == ["The", "cat", "sat"]
    assert all(interval.tier == "words" for interval in intervals)


def test_get_tier_nonexistent_returns_empty():
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    assert textgrid.get_tier("nonexistent") == []


def test_get_word_intervals_returns_words_tier():
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    intervals = textgrid.get_word_intervals()
    assert [interval.label for interval in intervals] == ["The", "cat", "sat"]


def test_get_phone_intervals_empty_without_phone_tier():
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    assert textgrid.get_phone_intervals() == []


def test_parse_from_string_captures_global_bounds():
    """``parse_from_string`` captures the global ``xmin``/``xmax`` from the
    Praat long-format header, setting ``start_time`` and ``end_time``.
    """
    textgrid = TextGridParser().parse_from_string(TEXTGRID_STRING)
    assert textgrid.start_time == 0.0
    assert textgrid.end_time == 3.0


# ---------------------------------------------------------------------------
# AlignmentHandler
# ---------------------------------------------------------------------------

def test_init_stores_sampling_rate():
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    assert handler.sampling_rate == 1000.0


def test_load_textgrid_from_string_returns_textgrid():
    handler = AlignmentHandler()
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    assert isinstance(textgrid, TextGrid)
    assert [interval.label for interval in textgrid.get_word_intervals()] == [
        "The",
        "cat",
        "sat",
    ]


def test_align_word_features_fills_each_word_interval():
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    word_features = {
        0: {"surprisal": 1.0},
        1: {"surprisal": 2.0},
        2: {"surprisal": 3.0},
    }
    aligned, feature_names = handler.align_word_features(
        word_features, textgrid, signal_length=3000
    )
    assert aligned.shape == (3000, 1)
    assert feature_names == ["surprisal"]  # sorted
    np.testing.assert_array_equal(aligned[:1000, 0], np.full(1000, 1.0))
    np.testing.assert_array_equal(aligned[1000:2000, 0], np.full(1000, 2.0))
    np.testing.assert_array_equal(aligned[2000:3000, 0], np.full(1000, 3.0))


def test_align_word_features_empty_word_features():
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    aligned, feature_names = handler.align_word_features(
        {}, textgrid, signal_length=3000
    )
    assert isinstance(aligned, np.ndarray)
    assert aligned.size == 0
    assert feature_names == []


def test_align_word_features_missing_word_index_stays_zero():
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    word_features = {
        0: {"surprisal": 1.0},
        2: {"surprisal": 3.0},  # word index 1 intentionally missing
    }
    aligned, feature_names = handler.align_word_features(
        word_features, textgrid, signal_length=3000
    )
    assert aligned.shape == (3000, 1)
    np.testing.assert_array_equal(aligned[:1000, 0], np.full(1000, 1.0))
    np.testing.assert_array_equal(aligned[1000:2000, 0], np.zeros(1000))
    np.testing.assert_array_equal(aligned[2000:3000, 0], np.full(1000, 3.0))


def test_align_word_features_signal_length_none_uses_end_time():
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    word_features = {
        0: {"surprisal": 1.0},
        1: {"surprisal": 2.0},
        2: {"surprisal": 3.0},
    }

    # end_time is parsed from the TextGrid global xmax (= 3.0), so the
    # derived signal length is 3000 samples at 1000 Hz.
    aligned, feature_names = handler.align_word_features(
        word_features, textgrid, signal_length=None
    )
    assert aligned.shape == (3000, 1)
    assert feature_names == ["surprisal"]
    np.testing.assert_array_equal(aligned[:1000, 0], np.full(1000, 1.0))
    np.testing.assert_array_equal(aligned[1000:2000, 0], np.full(1000, 2.0))
    np.testing.assert_array_equal(aligned[2000:3000, 0], np.full(1000, 3.0))


# ---------------------------------------------------------------------------
# Empty labels and compact TextGrid format
# ---------------------------------------------------------------------------

TEXTGRID_WITH_EMPTY = '''File type = "ooTextFile"
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
            text = ""
        intervals [3]:
            xmin = 2
            xmax = 3
            text = "sat"
'''

COMPACT_TEXTGRID = '''item [1]: "words"
intervals [1]: "hello" 0.0 1.0
intervals [2]: "world" 1.0 2.0
'''


def test_parse_empty_label_interval_is_kept():
    """An interval with an empty label (``text = ""``) is kept, not dropped.

    The label is preserved as an empty string so the interval count and the
    positions of later intervals stay aligned with the file.
    """
    textgrid = TextGridParser().parse_from_string(TEXTGRID_WITH_EMPTY)
    intervals = textgrid.get_word_intervals()
    assert len(intervals) == 3
    labels = [interval.label for interval in intervals]
    assert labels == ["The", "", "sat"]
    assert intervals[1].label == ""
    assert intervals[1].start == 1.0
    assert intervals[1].end == 2.0


def test_parse_compact_textgrid():
    """Compact-format TextGrids parse with correct labels and times."""
    textgrid = TextGridParser().parse_from_string(COMPACT_TEXTGRID)
    intervals = textgrid.get_word_intervals()
    assert len(intervals) == 2
    assert [interval.label for interval in intervals] == ["hello", "world"]
    assert [(interval.start, interval.end) for interval in intervals] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]


def test_compact_textgrid_end_time_can_be_set():
    """Compact format works with ``end_time`` set manually.

    Compact-format TextGrids have no global ``xmin``/``xmax`` header lines, so
    ``end_time`` defaults to 0.0; setting it explicitly lets
    :meth:`AlignmentHandler.align_word_features` derive ``signal_length``.
    """
    textgrid = TextGridParser().parse_from_string(COMPACT_TEXTGRID)
    assert textgrid.start_time == 0.0
    assert textgrid.end_time == 0.0

    textgrid.end_time = 2.0
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    word_features = {
        0: {"surprisal": 1.0},
        1: {"surprisal": 2.0},
    }
    aligned, feature_names = handler.align_word_features(
        word_features, textgrid, signal_length=None
    )
    assert aligned.shape == (2000, 1)
    assert feature_names == ["surprisal"]
    np.testing.assert_array_equal(aligned[:1000, 0], np.full(1000, 1.0))
    np.testing.assert_array_equal(aligned[1000:2000, 0], np.full(1000, 2.0))


def test_align_two_features_value_level():
    """Multi-feature alignment fills both feature columns per word interval.

    Verifies the returned shape, the alphabetically sorted feature names and
    the specific values present in each word's sample range for both features.
    """
    handler = AlignmentHandler(signal_sampling_rate=1000.0)
    textgrid = handler.load_textgrid_from_string(TEXTGRID_STRING)
    word_features = {
        0: {"surprisal": 1.0, "entropy": 0.5},
        1: {"surprisal": 2.0, "entropy": 1.5},
        2: {"surprisal": 3.0, "entropy": 2.5},
    }
    aligned, feature_names = handler.align_word_features(
        word_features, textgrid, signal_length=3000
    )
    assert aligned.shape == (3000, 2)
    assert feature_names == ["entropy", "surprisal"]  # sorted

    # Column 0 is "entropy", column 1 is "surprisal".
    np.testing.assert_array_equal(aligned[:1000, 0], np.full(1000, 0.5))
    np.testing.assert_array_equal(aligned[1000:2000, 0], np.full(1000, 1.5))
    np.testing.assert_array_equal(aligned[2000:3000, 0], np.full(1000, 2.5))
    np.testing.assert_array_equal(aligned[:1000, 1], np.full(1000, 1.0))
    np.testing.assert_array_equal(aligned[1000:2000, 1], np.full(1000, 2.0))
    np.testing.assert_array_equal(aligned[2000:3000, 1], np.full(1000, 3.0))