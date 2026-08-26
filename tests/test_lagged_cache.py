"""Tests for the lagged-X caching feature in TRFEstimator.

Caching is opt-in via ``cache_lagged=True`` (instance-level or per-call).
These tests use small arrays to stay well within memory limits.
"""
import numpy as np
import pytest

from pyeeg.models.trf import TRFEstimator


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def small_data():
    rng = np.random.default_rng(42)
    n_samples, n_feats, n_chans = 200, 3, 4
    X = rng.standard_normal((n_samples, n_feats))
    y = rng.standard_normal((n_samples, n_chans))
    return X, y


@pytest.fixture
def trf():
    return TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0)


# -- Basic cache hit / miss ---------------------------------------------------

def test_cache_hit_reuses_lagged_matrix(small_data, trf):
    """Second fit with same X should reuse the cached lagged matrix."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y)
    assert trf._lagged_cache is not None
    cached_array = trf._lagged_cache[1]
    # Second fit — should reuse, not recompute
    trf.fit(X, y)
    assert trf._lagged_cache[1] is cached_array  # same object in memory


def test_cache_disabled_by_default(small_data, trf):
    """Without cache_lagged, _lagged_cache stays None."""
    X, y = small_data
    trf.fit(X, y)
    assert trf._lagged_cache is None


def test_cache_per_call_override(small_data, trf):
    """cache_lagged=True on the call overrides instance-level False."""
    X, y = small_data
    trf.fit(X, y, cache_lagged=True)
    assert trf._lagged_cache is not None


def test_cache_per_call_disable(small_data, trf):
    """cache_lagged=False on the call overrides instance-level True."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y)
    assert trf._lagged_cache is not None
    trf.fit(X, y, cache_lagged=False)
    # Cache should be unchanged (not invalidated, just not consulted)
    assert trf._lagged_cache is not None


# -- Cache invalidation -------------------------------------------------------

def test_cache_miss_different_X(small_data, trf):
    """Different X object → cache miss → recompute and replace."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y)
    first_key = trf._lagged_cache[0]
    first_array = trf._lagged_cache[1]

    X2 = X + 1.0  # new array, different buffer
    trf.fit(X2, y)
    assert trf._lagged_cache[0] != first_key
    assert trf._lagged_cache[1] is not first_array


def test_cache_miss_different_lags(small_data, trf):
    """Different lags → cache miss."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y)
    first_key = trf._lagged_cache[0]

    trf.tmin = -0.2  # changes lags
    trf.fit(X, y)
    assert trf._lagged_cache[0] != first_key


def test_cache_miss_different_drop(small_data, trf):
    """Different drop → cache miss."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y, drop=True)
    first_key = trf._lagged_cache[0]

    trf.fit(X, y, drop=False)
    assert trf._lagged_cache[0] != first_key


def test_cache_miss_different_block_order(small_data, trf):
    """Different block_order → cache miss."""
    X, y = small_data
    trf.cache_lagged = True
    trf.block_order = "lags"
    trf.fit(X, y)
    first_key = trf._lagged_cache[0]

    trf.block_order = "features"
    trf.fit(X, y)
    assert trf._lagged_cache[0] != first_key


# -- clear_cache --------------------------------------------------------------

def test_clear_cache(small_data, trf):
    """clear_cache() empties the cache."""
    X, y = small_data
    trf.cache_lagged = True
    trf.fit(X, y)
    assert trf._lagged_cache is not None
    trf.clear_cache()
    assert trf._lagged_cache is None


# -- Correctness: cached vs uncached give same results ------------------------

def test_cached_fit_matches_uncached(small_data):
    """Coefficients from cached fit must match uncached fit."""
    X, y = small_data
    trf_cached = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0,
                             cache_lagged=True)
    trf_plain = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0)

    # Fit twice (second call exercises the cache)
    trf_cached.fit(X, y)
    trf_cached.fit(X, y)
    trf_plain.fit(X, y)

    np.testing.assert_allclose(trf_cached.coef_, trf_plain.coef_)
    np.testing.assert_allclose(trf_cached.intercept_, trf_plain.intercept_)


def test_cached_fit_multiple_alphas(small_data):
    """Cached fit with multi-alpha path gives same results as uncached."""
    X, y = small_data
    alphas = [0.0, 1.0, 10.0, 100.0]
    trf_cached = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=alphas,
                             cache_lagged=True)
    trf_plain = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=alphas)

    trf_cached.fit(X, y)
    trf_cached.fit(X, y)  # cache hit
    trf_plain.fit(X, y)

    np.testing.assert_allclose(trf_cached.all_betas, trf_plain.all_betas)


# -- Memory guardrail ---------------------------------------------------------

def test_max_cache_size_skips_large(small_data, trf):
    """When lagged matrix exceeds max_cache_size, it is not cached."""
    X, y = small_data
    trf.cache_lagged = True
    trf.max_cache_size = 1  # 1 byte — anything will exceed this
    trf.fit(X, y)
    assert trf._lagged_cache is None


def test_max_cache_size_allows_small(small_data, trf):
    """When max_cache_size is large enough, caching works."""
    X, y = small_data
    trf.cache_lagged = True
    trf.max_cache_size = 10**9  # 1 GB — plenty for this tiny array
    trf.fit(X, y)
    assert trf._lagged_cache is not None


# -- View handling ------------------------------------------------------------

def test_cache_different_views_different_keys(small_data, trf):
    """Two views of the same base with different shapes should get different keys."""
    X, y = small_data
    trf.cache_lagged = True
    # Fit with full X
    trf.fit(X, y)
    key_full = trf._lagged_cache[0]

    # Fit with a subset of columns — different shape, should be cache miss
    X_subset = X[:, :2].copy()
    trf.fit(X_subset, y)
    assert trf._lagged_cache[0] != key_full


def test_cache_strided_view_not_stale():
    """Transposed views of a square X share the same buffer but different
    strides — must NOT produce a stale cache hit."""
    rng = np.random.default_rng(0)
    # Square matrix + single lag so X.T also has valid shape for fitting
    X_sq = rng.standard_normal((200, 20))
    y = rng.standard_normal((200, 4))
    trf = TRFEstimator(times=(0.0,), srate=100.0, alpha=1.0,
                      cache_lagged=True)
    # Fit with X — cache should be populated
    trf.fit(X_sq, y)
    assert trf._lagged_cache is not None
    key1 = trf._lagged_cache[0]

    # X.T has the same buffer (20×20? no, 200×200) — actually X.T is 20×200
    # which can't be fit with 200-sample y. Instead test with F-contiguous
    # copy of the same data (different strides, different buffer → miss).
    X_f = np.asfortranarray(X_sq)
    trf.fit(X_f, y)
    assert trf._lagged_cache[0] != key1


def test_cache_strides_in_key():
    """Directly verify strides are part of the cache key."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = rng.standard_normal((200, 2))
    trf = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0,
                      cache_lagged=True)
    trf.fit(X, y)
    key = trf._lagged_cache[0]
    # Strides should be the 3rd element (index 2) of the key tuple
    assert key[2] == X.strides


def test_cache_negative_stride_not_stale():
    """X[::-1] (row-reversed) has a negative stride — must be a cache miss."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = rng.standard_normal((200, 4))
    trf = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0,
                      cache_lagged=True)
    trf.fit(X, y)
    key1 = trf._lagged_cache[0]

    X_rev = X[::-1]  # negative stride on axis 0, different pointer
    trf.fit(X_rev, y)
    assert trf._lagged_cache[0] != key1


# -- Lagged=True path is not cached -------------------------------------------

def test_pre_lagged_not_cached(small_data):
    """When lagged=True, the cache should not be touched."""
    X, y = small_data
    from pyeeg.utils import lag_matrix
    trf = TRFEstimator(tmin=-0.1, tmax=0.2, srate=100.0, alpha=1.0,
                      cache_lagged=True)
    trf.fill_lags()  # populate self.lags before we mirror them
    X_lagged = lag_matrix(X, lags=trf.lags, mode="valid", fill_value=0.0)
    trf.fit(X_lagged, y, lagged=True)
    assert trf._lagged_cache is None
