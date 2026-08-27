"""Tests for the syntactic feature extraction module.

The tree-based methods of :class:`SyntacticFeatureExtractor` work without an
external parser or network, operating directly on ``nltk.Tree`` objects. The
text-level entry points (``extract_to_dict`` / ``extract_to_array``) call
``parse_text`` internally, so those tests mock ``parse_text`` to return
pre-built trees.
"""

import numpy as np
import pytest

nltk = pytest.importorskip("nltk")

from pyeeg.features.syntactic_features import ParserConfig, SyntacticFeatureExtractor


def _make_extractor():
    return SyntacticFeatureExtractor()


def _sample_tree():
    """A small hand-built constituency tree."""
    return nltk.Tree.fromstring("(S (NP (D the) (N cat)) (VP (V sat)))")


def _two_trees():
    """Two trees whose leaf counts match ``SAMPLE_TEXT.split()``."""
    t1 = nltk.Tree.fromstring("(S (NP (D the) (N cat)) (VP (V sat)))")
    t2 = nltk.Tree.fromstring("(S (NP (D a) (N dog)))")
    return [t1, t2]


# ---------------------------------------------------------------------------
# ParserConfig defaults
# ---------------------------------------------------------------------------

def test_parser_config_defaults():
    config = ParserConfig()
    assert config.parser_name == "stanford"
    assert config.parser_path is None
    assert config.model_path is None
    assert config.language == "en"
    assert config.timeout == 30


# ---------------------------------------------------------------------------
# depth_single_tree
# ---------------------------------------------------------------------------

def test_depth_single_tree_returns_one_value_per_leaf():
    x = _make_extractor()
    depths = x.depth_single_tree(_sample_tree())
    assert isinstance(depths, list)
    assert len(depths) == len(_sample_tree().leaves()) == 3
    assert all(isinstance(d, int) for d in depths)
    # All depths are positive integers when the root S is discounted.
    assert all(d > 0 for d in depths)
    assert depths == [1, 1, 1]


def test_depth_single_tree_remove_S_offsets_by_one():
    x = _make_extractor()
    with_s = x.depth_single_tree(_sample_tree(), remove_S=False)
    without_s = x.depth_single_tree(_sample_tree(), remove_S=True)
    assert [d - 1 for d in with_s] == without_s


# ---------------------------------------------------------------------------
# opening_single_tree / closing_single_tree
# ---------------------------------------------------------------------------

def test_opening_single_tree_returns_one_value_per_leaf():
    x = _make_extractor()
    opening = x.opening_single_tree(_sample_tree())
    assert isinstance(opening, list)
    assert len(opening) == len(_sample_tree().leaves())
    assert all(isinstance(v, int) for v in opening)
    assert opening == [2, 0, 1]


def test_closing_single_tree_returns_one_value_per_leaf():
    x = _make_extractor()
    closing = x.closing_single_tree(_sample_tree())
    assert isinstance(closing, list)
    assert len(closing) == len(_sample_tree().leaves())
    assert all(isinstance(v, int) for v in closing)
    assert closing == [0, 1, 2]


# ---------------------------------------------------------------------------
# extract_from_tree
# ---------------------------------------------------------------------------

def test_extract_from_tree_named_features():
    x = _make_extractor()
    result = x.extract_from_tree(
        _sample_tree(), ["depth", "opening", "closing", "tree_height"])
    assert set(result.keys()) == {"depth", "opening", "closing", "tree_height"}
    n_leaves = len(_sample_tree().leaves())
    for values in result.values():
        assert isinstance(values, list)
        assert len(values) == n_leaves
        assert all(isinstance(v, int) for v in values)


def test_extract_from_tree_all_features():
    x = _make_extractor()
    result = x.extract_from_tree(_sample_tree(), ["all"])
    assert set(result.keys()) == {"depth", "opening", "closing", "tree_height"}
    n_leaves = len(_sample_tree().leaves())
    for values in result.values():
        assert len(values) == n_leaves
    assert result["depth"] == [1, 1, 1]
    assert result["opening"] == [2, 0, 1]
    assert result["closing"] == [0, 1, 2]


def test_extract_from_tree_tree_height_is_constant():
    x = _make_extractor()
    tree = _sample_tree()
    result = x.extract_from_tree(tree, ["tree_height"])
    n_leaves = len(tree.leaves())
    assert result["tree_height"] == [tree.height()] * n_leaves


def test_extract_from_tree_ignores_unknown_features():
    x = _make_extractor()
    result = x.extract_from_tree(_sample_tree(), ["depth", "not_a_feature"])
    assert set(result.keys()) == {"depth"}
    result_all = x.extract_from_tree(_sample_tree(), ["all", "not_a_feature"])
    assert set(result_all.keys()) == {"depth", "opening", "closing", "tree_height"}


# ---------------------------------------------------------------------------
# extract_to_dict (mocked parse_text)
# ---------------------------------------------------------------------------

def test_extract_to_dict_mocked_parse_text(monkeypatch):
    x = _make_extractor()
    trees = _two_trees()
    monkeypatch.setattr(x, "parse_text", lambda text: trees)

    text = "the cat sat a dog"
    result = x.extract_to_dict(text, features=["all"])

    # One entry per word position, per feature.
    assert set(result.keys()) == {"depth", "opening", "closing", "tree_height"}
    n_words = len(text.split())
    for feat_values in result.values():
        assert isinstance(feat_values, dict)
        assert len(feat_values) == n_words
        assert set(feat_values.keys()) == set(range(n_words))
        for value in feat_values.values():
            assert isinstance(value, float)

    # Concatenated across both trees in reading order.
    assert result["depth"] == {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}
    assert result["opening"] == {0: 2.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0}
    assert result["closing"] == {0: 0.0, 1: 1.0, 2: 2.0, 3: 0.0, 4: 1.0}
    # Heights: 4 for the first tree, 4 for the second.
    assert result["tree_height"] == {0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0}


def test_extract_to_dict_with_named_features(monkeypatch):
    x = _make_extractor()
    monkeypatch.setattr(x, "parse_text", lambda text: _two_trees())
    result = x.extract_to_dict("the cat sat a dog", features=["depth"])
    assert set(result.keys()) == {"depth"}
    assert result["depth"] == {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}


# ---------------------------------------------------------------------------
# extract_to_array (mocked parse_text)
# ---------------------------------------------------------------------------

def test_extract_to_array_mocked_parse_text(monkeypatch):
    x = _make_extractor()
    text = "the cat sat a dog"
    trees = _two_trees()
    monkeypatch.setattr(x, "parse_text", lambda t: trees)

    words, array = x.extract_to_array(text, features=None)

    assert isinstance(words, list)
    assert words == text.split()
    assert isinstance(array, np.ndarray)
    n_words = len(words)
    assert array.shape == (n_words, 4)
    np.testing.assert_array_equal(
        array[:, 0], [1.0, 1.0, 1.0, 0.0, 0.0])  # depth
    np.testing.assert_array_equal(
        array[:, 1], [2.0, 0.0, 1.0, 1.0, 0.0])  # opening
    np.testing.assert_array_equal(
        array[:, 2], [0.0, 1.0, 2.0, 0.0, 1.0])  # closing
    np.testing.assert_array_equal(
        array[:, 3], [4.0, 4.0, 4.0, 4.0, 4.0])  # tree_height


def test_extract_to_array_feature_column_order(monkeypatch):
    x = _make_extractor()
    text = "the cat sat a dog"
    monkeypatch.setattr(x, "parse_text", lambda t: _two_trees())

    # features=None defaults to ["all"] -> canonical column order.
    _, array_default = x.extract_to_array(text, features=None)
    assert array_default.shape[1] == 4

    # features containing "all" -> canonical column order.
    _, array_all = x.extract_to_array(text, features=["all"])
    assert array_all.shape[1] == 4
    np.testing.assert_array_equal(array_default, array_all)

    # Explicit named features follow the requested order.
    _, array_named = x.extract_to_array(text, features=["tree_height", "depth"])
    assert array_named.shape[1] == 2
    np.testing.assert_array_equal(array_named[:, 0], [4.0, 4.0, 4.0, 4.0, 4.0])
    np.testing.assert_array_equal(array_named[:, 1], [1.0, 1.0, 1.0, 0.0, 0.0])