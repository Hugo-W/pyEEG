"""Tests for the feature dimensionality reduction module (``pyeeg.features.reduction``)."""

import numpy as np
import pytest

from pyeeg.features.reduction import FeatureReducer, ReductionConfig


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _synthetic_features(n_samples=100, n_features=5, seed=42):
    """Return a 2D array with clear variance structure.

    Two latent directions with large variance plus small noise so that the
    first two principal components capture >95% of the total variance.
    """
    rng = np.random.default_rng(seed)
    loadings = np.array([
        [3.0, 0.2, 0.1, 0.05, 0.0],
        [0.2, 2.5, 0.15, 0.0, 0.05],
    ])
    base = rng.standard_normal((n_samples, 2)) @ loadings
    noise = 0.1 * rng.standard_normal((n_samples, n_features))
    return base + noise


# ---------------------------------------------------------------------------
# ReductionConfig defaults
# ---------------------------------------------------------------------------

def test_reduction_config_defaults():
    config = ReductionConfig()
    assert config.method == "pca"
    assert config.n_components is None
    assert config.variance_threshold == 0.95
    assert config.random_state is None
    assert config.whiten is False


# ---------------------------------------------------------------------------
# method="none"
# ---------------------------------------------------------------------------

def test_none_fit_transform_returns_input_unchanged():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="none"))
    np.testing.assert_array_equal(reducer.fit(X), reducer)
    np.testing.assert_array_equal(reducer.transform(X), X)
    np.testing.assert_array_equal(reducer.fit_transform(X), X)


def test_none_inverse_transform_returns_input_unchanged():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="none"))
    reducer.fit(X)
    np.testing.assert_array_equal(reducer.inverse_transform(X), X)


def test_none_getters_return_none():
    reducer = FeatureReducer(ReductionConfig(method="none"))
    reducer.fit(_synthetic_features())
    assert reducer.get_explained_variance() is None
    assert reducer.get_components() is None
    assert reducer.get_n_components() is None


# ---------------------------------------------------------------------------
# method="pca"
# ---------------------------------------------------------------------------

def test_pca_fit_sets_fitted_and_returns_self():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca"))
    assert reducer._fitted is False
    result = reducer.fit(X)
    assert result is reducer
    assert reducer._fitted is True


def test_pca_transform_shape_from_variance_threshold():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", variance_threshold=0.95))
    transformed = reducer.fit_transform(X)
    n_components = reducer.get_n_components()
    assert transformed.shape == (X.shape[0], n_components)
    assert transformed.shape[1] >= 1
    assert 1 <= n_components <= X.shape[1]


def test_pca_explained_variance_is_1d_ratios():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", variance_threshold=0.95))
    reducer.fit(X)
    explained = reducer.get_explained_variance()
    assert explained is not None
    assert explained.ndim == 1
    assert explained.shape == (X.shape[1],)
    assert np.all(explained >= 0.0)
    assert np.isclose(explained.sum(), 1.0)
    # First two components capture >95% of the variance for this dataset.
    assert np.cumsum(explained)[reducer.get_n_components() - 1] >= 0.95


def test_pca_n_components_is_int():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", variance_threshold=0.95))
    reducer.fit(X)
    n_components = reducer.get_n_components()
    assert isinstance(n_components, (int, np.integer))
    assert int(n_components) >= 1


def test_pca_components_shape():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", variance_threshold=0.95))
    reducer.fit(X)
    components = reducer.get_components()
    assert components is not None
    assert components.ndim == 2
    assert components.shape == (reducer.get_n_components(), X.shape[1])


def test_pca_explicit_n_components_shape():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", n_components=2))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (X.shape[0], 2)
    assert reducer.get_n_components() == 2


def test_pca_inverse_transform_reconstructs_centered_input():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", n_components=2))
    transformed = reducer.fit_transform(X)
    reconstructed = reducer.inverse_transform(transformed)
    centered = X - X.mean(axis=0)
    np.testing.assert_allclose(reconstructed, centered, atol=1.0)


def test_pca_fit_transform_matches_fit_then_transform():
    X = _synthetic_features()
    reducer_a = FeatureReducer(ReductionConfig(method="pca", n_components=2))
    reducer_b = FeatureReducer(ReductionConfig(method="pca", n_components=2))
    combined = reducer_a.fit_transform(X)
    reducer_b.fit(X)
    separate = reducer_b.transform(X)
    np.testing.assert_allclose(combined, separate)


# ---------------------------------------------------------------------------
# method="ica"
# ---------------------------------------------------------------------------

def test_ica_fit_transform_shape():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="ica"))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (X.shape[0], X.shape[1])
    assert reducer.get_n_components() == X.shape[1]


def test_ica_explicit_n_components_shape():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="ica", n_components=2))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (X.shape[0], 2)
    assert reducer.get_n_components() == 2


def test_ica_n_components_is_int():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="ica"))
    reducer.fit(X)
    n_components = reducer.get_n_components()
    assert isinstance(n_components, (int, np.integer))
    assert int(n_components) == X.shape[1]


def test_ica_components_shape():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="ica"))
    reducer.fit(X)
    components = reducer.get_components()
    assert components is not None
    assert components.shape == (reducer.get_n_components(), X.shape[1])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_fit_rejects_empty_array():
    reducer = FeatureReducer(ReductionConfig())
    with pytest.raises(ValueError):
        reducer.fit(np.array([]))


def test_fit_rejects_1d_array():
    reducer = FeatureReducer(ReductionConfig())
    with pytest.raises(ValueError):
        reducer.fit(np.array([1.0, 2.0, 3.0]))


def test_transform_before_fit_raises():
    reducer = FeatureReducer(ReductionConfig())
    with pytest.raises(RuntimeError, match="Reducer not fitted"):
        reducer.transform(_synthetic_features())


def test_inverse_transform_before_fit_raises():
    reducer = FeatureReducer(ReductionConfig())
    with pytest.raises(RuntimeError, match="Reducer not fitted"):
        reducer.inverse_transform(_synthetic_features())


def test_unknown_method_raises():
    reducer = FeatureReducer(ReductionConfig(method="svd"))
    with pytest.raises(ValueError, match="Unknown reduction method"):
        reducer.fit(_synthetic_features())


# ---------------------------------------------------------------------------
# Round-trip reconstruction (pseudoinverse fix for ICA)
# ---------------------------------------------------------------------------

def test_ica_full_rank_round_trip_reconstruction():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    reducer = FeatureReducer(ReductionConfig(method="ica", n_components=5))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (100, 5)
    reconstructed = reducer.inverse_transform(transformed)
    # Full-rank whitening round trip should recover the input. This validates
    # the pseudoinverse used by ICA's inverse_transform.
    np.testing.assert_allclose(reconstructed, X, atol=1e-6)


def test_ica_reduced_round_trip_is_finite_with_reasonable_mse():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    reducer = FeatureReducer(ReductionConfig(method="ica", n_components=3))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (100, 3)
    reconstructed = reducer.inverse_transform(transformed)
    assert reconstructed.shape == (100, 5)
    assert np.all(np.isfinite(reconstructed))
    # Lossy (3 of 5 components) reconstruction should stay close for data
    # that is roughly standardized (standard normals here).
    mse = np.mean((reconstructed - X) ** 2)
    assert mse < 1.0


def test_pca_reduced_round_trip_reconstructs_approx():
    X = _synthetic_features()
    reducer = FeatureReducer(ReductionConfig(method="pca", n_components=3))
    transformed = reducer.fit_transform(X)
    assert transformed.shape == (X.shape[0], 3)
    reconstructed = reducer.inverse_transform(transformed)
    # The first three PCs capture >99% of this dataset's variance, so the
    # lossy projection should closely approximate the original.
    np.testing.assert_allclose(reconstructed, X, atol=0.5)