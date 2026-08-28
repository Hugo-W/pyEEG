"""Statistical inference for TRF analysis.

This module provides nonparametric statistical methods for temporal response
function (TRF) analysis, complementing the parametric ``tvals_``/``pvals_``
path in :class:`pyeeg.models.TRFEstimator`.

The primary methods are:

- :func:`permutation_test_trf`: circular-shift permutation test with FWE
  correction via the max-statistic.
- :func:`cluster_based_permutation_test`: cluster-based correction
  (Maris & Oostenveld 2007) on top of the permutation engine.
- :func:`bootstrap_ci_trf`: paired block-bootstrap confidence intervals.
- :func:`cross_subject_consistency`: descriptive cross-subject reliability.
- :func:`group_level_test`: sign-flip group-level inference on coefficient
  maps.

The default statistic is ``stat="zscore"``: the stats function internally
z-scores each input feature of ``X`` and each channel of ``y`` *before* lag
construction and fitting, producing a scale-standardised coefficient that is
mathematically clean for any solver (OLS, ridge, banded ridge, robust).
This is *not* a t-statistic; it removes measurement units but does not
equalise coefficient uncertainty.

No MNE imports occur in this module.  Spatial adjacency matrices must be
supplied by the user (e.g. from ``mne.channels.find_ch_adjacency``).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

import numpy as np

from pyeeg._logging import LOGGER

__all__ = [
    "permutation_test_trf",
    "cluster_based_permutation_test",
    "bootstrap_ci_trf",
    "cross_subject_consistency",
    "group_level_test",
    "jackknife_se_trf",
    "TRFAnalyzer",
    "PermutationResult",
    "ClusterResult",
    "BootstrapResult",
    "ConsistencyResult",
    "GroupTestResult",
    "JackknifeResult",
]


# ===========================================================================
# Result dataclasses
# ===========================================================================


@dataclass
class PermutationResult:
    """Result of a permutation test on TRF coefficients.

    Attributes
    ----------
    observed : ndarray, shape (n_lags, n_feats, n_chans)
        Observed statistic map.
    null_distribution : ndarray, shape (n_perm,)
        Max-statistic from each permutation (used for FWE correction).
    pvals_corrected : ndarray, shape (n_lags, n_feats, n_chans)
        FWE-corrected p-values (plus-one formula).
    stat : str
        Statistic used (``"zscore"``, ``"t"``, ``"coef"``, ``"perm_norm"``,
        ``"jackknife"``).
    tails : str
        Tail (``"two-sided"``, ``"positive"``, ``"negative"``).
    family : str
        Multiplicity family.
    n_perm : int
        Number of permutations.
    seed : int or None
        Random seed.
    hypothesis : str
        Null hypothesis description.
    resampling_scheme : str
        Resampling scheme description.
    zero_var_features : ndarray or None, shape (n_feats,)
        Boolean mask of zero-variance features (zscore stat only).
    zero_var_channels : ndarray or None, shape (n_chans,)
        Boolean mask of zero-variance channels (zscore stat only).
    zero_var_stat : ndarray or None, shape (n_lags, n_feats, n_chans)
        Boolean mask of zero-variance statistic coordinates (perm_norm only).
    """

    observed: np.ndarray
    null_distribution: np.ndarray
    pvals_corrected: np.ndarray
    stat: str
    tails: str
    family: str
    n_perm: int
    seed: Optional[int]
    hypothesis: str
    resampling_scheme: str
    zero_var_features: Optional[np.ndarray] = None
    zero_var_channels: Optional[np.ndarray] = None
    zero_var_stat: Optional[np.ndarray] = None


@dataclass
class ClusterResult:
    """Result of a cluster-based permutation test.

    Attributes
    ----------
    clusters : list of tuple
        Each tuple is ``(sign, indices, mass)`` where sign is +1 or -1,
        indices is a 1-D array of flattened coefficient indices, and mass
        is the cluster mass (sum of signed statistic).
    pvals : ndarray
        Per-cluster p-values (plus-one formula).
    mask_significant : ndarray, shape (n_lags, n_feats, n_chans)
        Boolean mask of significant coefficients.
    threshold : float
        Cluster-forming threshold used.
    family : str
        Multiplicity family.
    n_perm : int
        Number of permutations.
    adjacency_layout : str
        Description of the adjacency layout.
    """

    clusters: list
    pvals: np.ndarray
    mask_significant: np.ndarray
    threshold: float
    family: str
    n_perm: int
    adjacency_layout: str


@dataclass
class BootstrapResult:
    """Result of a block-bootstrap confidence interval estimation.

    Attributes
    ----------
    ci_low : ndarray, shape (n_lags, n_feats, n_chans)
        Lower bound of the confidence interval.
    ci_high : ndarray, shape (n_lags, n_feats, n_chans)
        Upper bound of the confidence interval.
    distribution : ndarray or None
        Full bootstrap distribution if ``return_distribution=True``,
        shape ``(n_boot, n_lags, n_feats, n_chans)``.
    se : ndarray, shape (n_lags, n_feats, n_chans)
        Bootstrap standard error (std of the distribution).
    block_size_ : int
        Block size used (estimated or user-supplied).
    method : str
        Bootstrap method (``"circular"``).
    n_boot : int
        Number of bootstrap replications.
    estimand : str
        Description of the estimand.
    """

    ci_low: np.ndarray
    ci_high: np.ndarray
    distribution: Optional[np.ndarray]
    se: np.ndarray
    block_size_: int
    method: str
    n_boot: int
    estimand: str


@dataclass
class ConsistencyResult:
    """Result of descriptive cross-subject consistency.

    Attributes
    ----------
    consistency : ndarray
        Per-(lag, feature) or per-(lag, feature, channel) consistency.
    per_subject : ndarray or None
        Per-subject reliability (LOO mode only).
    descriptive_only : bool
        Always True — no inferential test.
    """

    consistency: np.ndarray
    per_subject: Optional[np.ndarray]
    descriptive_only: bool = True


@dataclass
class GroupTestResult:
    """Result of a group-level sign-flip test.

    Attributes
    ----------
    mean_coef : ndarray, shape (n_lags, n_feats, n_chans)
        Group mean coefficient map.
    pvals_corrected : ndarray, shape (n_lags, n_feats, n_chans)
        FWE-corrected p-values (plus-one formula).
    mask_significant : ndarray, shape (n_lags, n_feats, n_chans)
        Boolean mask of significant coefficients.
    family : str
        Multiplicity family.
    n_perm : int
        Number of permutations.
    hypothesis : str
        Null hypothesis description.
    """

    mean_coef: np.ndarray
    pvals_corrected: np.ndarray
    mask_significant: np.ndarray
    family: str
    n_perm: int
    hypothesis: str


@dataclass
class JackknifeResult:
    """Result of a leave-one-epoch-out jackknife SE/CI estimation.

    This is a standalone uncertainty estimator (not a permutation test).
    It estimates the sampling variability of TRF coefficients by leaving
    one epoch out at a time and computing the jackknife standard error.

    Attributes
    ----------
    coef : ndarray, shape (n_lags, n_feats, n_chans)
        Full-data coefficient (from fitting all epochs).
    se : ndarray, shape (n_lags, n_feats, n_chans)
        Jackknife standard error.
    ci_low : ndarray, shape (n_lags, n_feats, n_chans)
        Lower CI bound (coef - z * se, z = 1.96 for 95%).
    ci_high : ndarray, shape (n_lags, n_feats, n_chans)
        Upper CI bound (coef + z * se).
    n_epochs : int
        Number of epochs used.
    alpha : float
        CI confidence level (1 - alpha).
    """

    coef: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_epochs: int
    alpha: float


# ===========================================================================
# Internal helpers
# ===========================================================================

_EPS = 1e-12


def _as_segment_list(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Normalise X/y to lists of 2-D segment arrays.

    Single 2-D arrays become one-element lists.  3-D y (n_epochs, n_samples,
    n_chans) is split into a list of (n_samples, n_chans) slices.
    """
    if isinstance(X, np.ndarray) and X.ndim == 2:
        X_list = [X]
    elif isinstance(X, (list, tuple)):
        X_list = list(X)
    else:
        raise TypeError("X must be a 2-D ndarray or a list of 2-D ndarrays.")

    if isinstance(y, np.ndarray) and y.ndim == 2:
        y_list = [y]
    elif isinstance(y, np.ndarray) and y.ndim == 3:
        # (n_epochs, n_samples, n_chans) -> list of (n_samples, n_chans)
        y_list = [y[i] for i in range(y.shape[0])]
    elif isinstance(y, (list, tuple)):
        y_list = list(y)
    else:
        raise TypeError("y must be a 2-D/3-D ndarray or a list of 2-D ndarrays.")

    if len(X_list) != len(y_list):
        raise ValueError(
            f"X has {len(X_list)} segments but y has {len(y_list)}."
        )
    return X_list, y_list


def _zscore_array(
    arr: np.ndarray, weights: Optional[np.ndarray] = None, eps: float = _EPS
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score a 2-D array along axis 0 (per column).

    Returns (zscored, zero_var_mask) where zero_var_mask is a boolean 1-D
    array marking columns with std < eps (set to zeros in the output).
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            return np.zeros_like(arr), np.ones(arr.shape[1], dtype=bool)
        mean = (w[:, None] * arr).sum(axis=0) / w_sum
        var = (w[:, None] * (arr - mean[None, :]) ** 2).sum(axis=0) / w_sum
    else:
        mean = arr.mean(axis=0)
        var = arr.var(axis=0)  # ddof=0

    std = np.sqrt(var)
    zero_var = std < eps
    std_safe = np.where(zero_var, 1.0, std)
    z = (arr - mean[None, :]) / std_safe[None, :]
    z[:, zero_var] = 0.0
    return z, zero_var


def _zscore_segments(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    weights: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    eps: float = _EPS,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Z-score each segment independently (pre-lag).

    Returns (X_z_list, y_z_list, zero_var_features, zero_var_channels).
    The zero-variance masks are the union across all segments.
    """
    n_feats = X_list[0].shape[1]
    n_chans = y_list[0].shape[1]
    zero_var_feats = np.zeros(n_feats, dtype=bool)
    zero_var_chans = np.zeros(n_chans, dtype=bool)

    X_z_list = []
    y_z_list = []

    for i, (Xi, yi) in enumerate(zip(X_list, y_list)):
        wi = weights[i] if isinstance(weights, (list, tuple)) else weights
        Xi_z, zvf = _zscore_array(Xi, wi, eps)
        yi_z, zvc = _zscore_array(yi, wi, eps)
        zero_var_feats |= zvf
        zero_var_chans |= zvc
        X_z_list.append(Xi_z)
        y_z_list.append(yi_z)

    return X_z_list, y_z_list, zero_var_feats, zero_var_chans


def _get_window_samples(trf) -> Tuple[int, int]:
    """Return (tmin_samp, tmax_samp) from a fitted/unfitted TRFEstimator.

    Uses trf.lags (descending samples) to derive the window extent.
    """
    if trf.lags is None:
        trf.fill_lags()
    lags = np.asarray(trf.lags)
    # lags are descending (e.g. [50, 49, ..., -20]); tmin_samp = min(lags),
    # tmax_samp = max(lags)
    tmin_samp = int(lags.min())
    tmax_samp = int(lags.max())
    return tmin_samp, tmax_samp


def _valid_offsets(
    n_samples: int,
    tmin_samp: int,
    tmax_samp: int,
    margin: Optional[int] = None,
) -> np.ndarray:
    """Compute the set of valid circular-shift offsets.

    A shift s is invalid if it could re-align a true relationship, i.e.
    s ∈ [tmin_samp - margin, tmax_samp + margin].  Valid offsets are all
    others modulo n_samples.

    Returns a 1-D int array of valid offsets.
    """
    window_extent = tmax_samp - tmin_samp
    if margin is None:
        margin = window_extent  # conservative

    # Invalid range (as continuous integers)
    lo = tmin_samp - margin
    hi = tmax_samp + margin

    all_offsets = np.arange(n_samples)
    # An offset s is invalid if (s mod n) falls in [lo, hi] (wrapping).
    # Since lo can be negative and hi can exceed n, we check membership
    # in the wrapped invalid set.
    invalid = np.zeros(n_samples, dtype=bool)
    for s in range(lo, hi + 1):
        invalid[s % n_samples] = True

    valid = all_offsets[~invalid]
    if len(valid) == 0:
        raise ValueError(
            f"No valid circular-shift offsets for n_samples={n_samples} "
            f"and lag window [{tmin_samp}, {tmax_samp}] (extent={window_extent}, "
            f"margin={margin}). Segment too short for the lag window."
        )
    return valid


def _circular_shift(
    X: np.ndarray, offset: int, axis: int = 0, fade_samples: int = 0
) -> np.ndarray:
    """Circularly shift X along axis by offset samples.

    If ``fade_samples > 0``, apply a raised-cosine (Tukey) taper at both
    edges of X *before* shifting so the wrap-around point is smooth.
    The taper ramps from 0 to 1 over ``fade_samples`` at each edge.
    """
    if fade_samples > 0:
        X = _fade_edges(X, fade_samples, axis=axis)
    return np.roll(X, offset, axis=axis)


def _fade_edges(
    X: np.ndarray, fade_samples: int, axis: int = 0
) -> np.ndarray:
    """Apply a raised-cosine taper at both edges of X along ``axis``.

    The taper ramps from 0 to 1 over ``fade_samples`` at the start and
    from 1 to 0 over ``fade_samples`` at the end, so that when the array
    is circularly shifted the wrap-around point is continuous.

    Parameters
    ----------
    X : ndarray
    fade_samples : int
        Number of samples to taper at each edge.
    axis : int
        Axis along which to apply the taper.

    Returns
    -------
    ndarray
        Tapered copy of X.
    """
    n = X.shape[axis]
    fade = min(fade_samples, n // 2)
    if fade <= 0:
        return X.copy()

    # Raised cosine: 0.5 * (1 - cos(pi * t / fade)) for t in [0, fade)
    t = np.arange(fade)
    ramp = 0.5 * (1.0 - np.cos(np.pi * t / fade))

    # Build the full window (1.0 in the middle, ramps at both ends)
    window = np.ones(n)
    window[:fade] = ramp
    window[-fade:] = ramp[::-1]

    # Broadcast window along the non-taper axes
    shape = [1] * X.ndim
    shape[axis] = n
    window = window.reshape(shape)

    return X * window


def _estimate_fade_samples(
    X: np.ndarray, srate: float, min_fade: int = 1, max_fade_frac: float = 0.25
) -> int:
    """Estimate the edge-fade length (in samples) from the stimulus spectrum.

    The fade must be long enough to smooth the wrap-around discontinuity for
    autocorrelated signals.  We use the signal's effective bandwidth: the
    -3 dB bandwidth (the range of frequencies within 3 dB of the peak power)
    gives a characteristic time scale ``1 / bandwidth`` over which the signal
    remains correlated.  The fade is set to this value, capped at
    ``max_fade_frac * n_samples``.

    For white noise (flat spectrum, full bandwidth) the fade will be
    minimal (~1 sample), which is correct since white noise has no
    autocorrelation to preserve.

    Parameters
    ----------
    X : ndarray, shape (n_samples,) or (n_samples, n_feats)
        Stimulus signal (one or more features).
    srate : float
        Sampling rate in Hz.
    min_fade : int
        Minimum fade length in samples.
    max_fade_frac : float
        Maximum fade length as a fraction of n_samples.

    Returns
    -------
    fade_samples : int
    """
    n = X.shape[0]
    max_fade = max(min_fade, int(n * max_fade_frac))

    if X.ndim == 1:
        X_2d = X[:, None]
    else:
        X_2d = X

    # Average periodogram across features
    psd = np.zeros(n // 2 + 1)
    for j in range(X_2d.shape[1]):
        xj = X_2d[:, j] - X_2d[:, j].mean()
        power = np.abs(np.fft.rfft(xj)) ** 2
        psd += power
    psd /= X_2d.shape[1]

    if psd.max() <= 0:
        return min_fade

    # -3 dB level: half the peak power (not the mean)
    peak_power = psd.max()
    threshold = peak_power / 2.0

    # Bandwidth: range of frequencies above -3 dB of peak
    above_3db = np.where(psd >= threshold)[0]
    if len(above_3db) <= 1:
        return min_fade

    freqs = np.fft.rfftfreq(n, d=1.0 / srate)
    bandwidth = freqs[above_3db[-1]] - freqs[above_3db[0]]

    if bandwidth <= 0:
        return min_fade

    # Characteristic time = 1 / bandwidth (time scale of amplitude changes)
    fade = int(np.ceil(srate / bandwidth))
    return max(min_fade, min(fade, max_fade))


def _clone_trf(trf, **overrides):
    """Clone a TRFEstimator configuration via copy() + attribute overrides.

    Returns a new unfitted estimator with the same config as ``trf`` but
    with selected attributes overridden.  The clone is ready for ``fit()``.
    """
    clone = trf.copy()
    clone.fitted = False
    clone.coef_ = None
    clone.intercept_ = None
    clone.tvals_ = None
    clone.pvals_ = None
    clone.standardized_coef_ = None
    clone.valid_samples_ = None
    clone.all_betas = None
    clone.robust_n_iter_ = None
    clone.robust_converged_ = None
    clone.robust_scale_ = None
    clone.robust_objective_ = None
    clone._lagged_cache = None
    for key, val in overrides.items():
        setattr(clone, key, val)
    return clone


def _fit_and_get_stat(
    trf_template,
    X,
    y,
    stat: str,
    weights=None,
    lagged: bool = False,
    zscored: bool = False,
    zero_var_info=None,
) -> np.ndarray:
    """Fit a clone of trf_template on (X, y) and return the statistic map.

    The statistic map has shape (n_lags, n_feats, n_chans).

    For stat="zscore": fit on z-scored data with fit_intercept=False,
    return coef_.
    For stat="coef": fit on original data, return coef_.
    For stat="t": fit on original data (OLS only), return tvals_ reshaped.
    """
    clone = _clone_trf(trf_template)

    if zscored:
        clone.fit_intercept = False

    # Fit
    if isinstance(X, list):
        clone.fit(X, y, lagged=lagged, weights=weights)
    else:
        clone.fit(X, y, lagged=lagged, weights=weights)

    if stat in ("zscore", "coef", "perm_norm"):
        stat_map = clone.coef_.copy()
    elif stat == "t":
        if clone.tvals_ is None:
            raise ValueError(
                "stat='t' requires OLS (alpha=None, loss='linear') but "
                "tvals_ was not computed. Check the TRF configuration."
            )
        n_lags = len(clone.lags)
        n_feats = clone.n_feats_
        n_chans = clone.n_chans_
        stat_map = clone.tvals_.reshape(n_lags, n_feats, n_chans).copy()
    else:
        raise ValueError(f"Unknown stat: {stat!r}")

    return stat_map


def _max_stat(stat_map: np.ndarray, tails: str, family_mask: np.ndarray) -> float:
    """Compute the max-statistic from a stat map, respecting tails and family.

    family_mask is a boolean array of the same shape as stat_map, selecting
    which elements belong to the FWE family.
    """
    masked = stat_map[family_mask]
    if tails == "two-sided":
        return float(np.max(np.abs(masked)))
    elif tails == "positive":
        return float(np.max(masked))
    elif tails == "negative":
        return float(np.min(masked))
    else:
        raise ValueError(f"Unknown tails: {tails!r}")


def _plus_one_pvals(
    observed_map: np.ndarray,
    null_max_stats: np.ndarray,
    tails: str,
    family_mask: np.ndarray,
) -> np.ndarray:
    """Compute plus-one FWE-corrected p-values.

    p_j = (1 + #{null_max >= |obs_j|}) / (1 + n_perm)

    Returns p-values in the same shape as observed_map (unmasked elements
    set to 1.0).
    """
    n_perm = len(null_max_stats)
    pvals = np.ones_like(observed_map, dtype=float)

    obs_vals = observed_map[family_mask]

    if tails == "two-sided":
        obs_abs = np.abs(obs_vals)
        null_abs = np.abs(null_max_stats)
    elif tails == "positive":
        obs_abs = obs_vals
        null_abs = null_max_stats
    elif tails == "negative":
        obs_abs = -obs_vals
        null_abs = -null_max_stats
    else:
        raise ValueError(f"Unknown tails: {tails!r}")

    # plus-one: p = (1 + #{null >= obs}) / (1 + n_perm)
    counts = np.sum(null_abs[None, :] >= obs_abs[:, None], axis=1)
    pvals_flat = (1 + counts) / (1 + n_perm)

    # Write back into the full-shaped array
    flat_idx = np.where(family_mask.ravel())[0]
    pvals_flat_full = np.ones(family_mask.size, dtype=float)
    pvals_flat_full[flat_idx] = pvals_flat
    pvals = pvals_flat_full.reshape(observed_map.shape)
    return pvals


# ===========================================================================
# Public API — permutation test
# ===========================================================================


def permutation_test_trf(
    trf,
    X,
    y,
    n_perm: int = 1000,
    stat: str = "zscore",
    tails: str = "two-sided",
    family: str = "global",
    n_jobs: int = 1,
    seed: Optional[int] = None,
    allow_robust: bool = False,
    weights=None,
    lagged: bool = False,
    fade_edges: bool = True,
    fade_time: Union[str, int] = "auto",
    verbose: bool = False,
) -> PermutationResult:
    """Permutation test for TRF coefficients using circular-shift null.

    Tests H0: no stimulus→response predictive relationship at any modeled
    lag.  The null is constructed by circularly shifting the stimulus X
    by a random valid offset (preserving autocorrelation, breaking
    X→y alignment), then refitting the TRF.

    Parameters
    ----------
    trf : TRFEstimator
        Fitted or configured TRF estimator. Its configuration (alpha,
        solver, loss, block_order, etc.) is cloned and frozen for all
        refits.
    X : ndarray or list of ndarray
        Stimulus features, shape (n_samples, n_feats) or list of segments.
        Must be raw (pre-lag) when ``stat="zscore"``.
    y : ndarray or list of ndarray
        Response, shape (n_samples, n_chans) or list of segments.
    n_perm : int, default 1000
        Number of permutations.
    stat : {"zscore", "t", "coef", "perm_norm", "jackknife"}
        Statistic to use. ``"zscore"`` (default) z-scores X and y before
        fitting, producing a scale-standardised coefficient clean for any
        solver. ``"t"`` uses parametric t-values (OLS only). ``"coef"``
        uses raw coefficients.
    tails : {"two-sided", "positive", "negative"}
        Tail of the test.
    family : {"global", "per_channel", "per_feature"} or ndarray
        Multiplicity family for FWE correction. ``"global"`` corrects over
        all (lag, feature, channel). A boolean ndarray mask of shape
        (n_lags, n_feats, n_chans) selects a custom family.
    n_jobs : int, default 1
        Number of parallel jobs (via joblib).
    seed : int or None
        Random seed for reproducibility.
    allow_robust : bool, default False
        If True, allow robust (Cauchy) loss refits (expensive). If False
        and the TRF has ``loss="cauchy"``, raises.
    weights : ndarray or list of ndarray, or None
        Sample weights, (n_samples,) or list per segment.
    lagged : bool, default False
        If True, X is a pre-lagged design. Rejected for ``stat="zscore"``,
        ``"perm_norm"``, ``"jackknife"``.
    fade_edges : bool, default True
        If True, apply a raised-cosine taper at both edges of each segment
        of X before circular shifting, smoothing the wrap-around
        discontinuity.  This is recommended for autocorrelated (smooth)
        stimuli, where a raw circular shift creates a spectral
        discontinuity that makes null coefficients artificially small.
        For white-noise stimuli the fade has minimal effect.
    fade_time : {"auto"} or int, default "auto"
        Number of samples to taper at each edge.  ``"auto"`` estimates the
        fade length from the stimulus spectrum: the highest frequency at
        -3 dB below mean power gives a characteristic time, and the fade
        is set to half a period of that frequency.  An explicit int
        overrides the automatic estimate.
    verbose : bool, default False

    Returns
    -------
    result : PermutationResult

    Notes
    -----
    This is a surrogate/randomization test under stationarity and
    alignment-exchangeability assumptions, not an exact finite-sample
    randomization test (the restricted offset set is not a permutation
    group). Circular-shift failure modes include nonstationarity, shared
    slow drift, and periodic stimuli.
    """
    from pyeeg.models import TRFEstimator

    # --- validate stat + lagged compatibility ---
    if stat in ("zscore", "perm_norm", "jackknife") and lagged:
        raise ValueError(
            f"stat={stat!r} requires raw (pre-lag) X; lagged=True is not "
            f"supported. Use stat='coef' or stat='t' for pre-lagged input."
        )

    # --- validate robust ---
    if trf.loss == "cauchy" and not allow_robust:
        raise ValueError(
            "Robust loss (Cauchy) refits are expensive. Set allow_robust=True "
            "to proceed, or use stat='coef' on a pre-fitted TRF."
        )

    # --- validate stat="t" requires OLS ---
    if stat == "t":
        if trf.alpha is not None and (np.isscalar(trf.alpha) and trf.alpha > 0):
            raise ValueError(
                "stat='t' requires OLS (alpha=None or alpha=0). "
                f"Got alpha={trf.alpha}."
            )
        if trf.loss not in ("linear", "none"):
            raise ValueError(
                f"stat='t' requires loss='linear', got loss={trf.loss!r}."
            )
        if trf.quadratic_reg is not None or trf.feature_alphas is not None:
            raise ValueError(
                "stat='t' requires OLS (no quadratic_reg or feature_alphas)."
            )

    # --- validate stat="jackknife" requires multi-epoch ---
    # (jackknife is a standalone SE estimator, not a permutation stat — see jackknife_se_trf)

    if stat not in ("zscore", "t", "coef", "perm_norm"):
        raise NotImplementedError(
            f"stat={stat!r} is not yet implemented. Use 'zscore', 't', 'coef', or 'perm_norm'."
        )

    # --- normalise segments ---
    X_list, y_list = _as_segment_list(X, y)
    n_segments = len(X_list)

    # --- weights to list ---
    if weights is not None and not isinstance(weights, (list, tuple)):
        weights_list = [weights] * n_segments if n_segments > 1 else [weights]
        if n_segments == 1:
            weights_list = [weights]
    elif weights is not None:
        weights_list = list(weights)
    else:
        weights_list = None

    # --- window samples for valid offsets ---
    tmin_samp, tmax_samp = _get_window_samples(trf)

    # --- check segment lengths ---
    for i, Xi in enumerate(X_list):
        _valid_offsets(len(Xi), tmin_samp, tmax_samp)  # raises if too short

    # --- estimate edge fade per segment ---
    fade_per_seg = [0] * n_segments
    if fade_edges:
        srate = trf.srate
        for i, Xi in enumerate(X_list):
            if isinstance(fade_time, str) and fade_time == "auto":
                fs = _estimate_fade_samples(Xi, srate)
            elif isinstance(fade_time, int):
                fs = fade_time
            else:
                raise ValueError(
                    f"fade_time must be 'auto' or an int, got {fade_time!r}"
                )
            fade_per_seg[i] = fs
            if verbose and fs > 0:
                LOGGER.info(
                    "Segment %d: fade_samples=%d (%.3f s)",
                    i, fs, fs / srate,
                )

    # --- standardisation (if zscore or perm_norm — perm_norm uses zscore as base) ---
    zscored = stat in ("zscore", "perm_norm")
    zero_var_feats = None
    zero_var_chans = None

    if zscored:
        X_list, y_list, zero_var_feats, zero_var_chans = _zscore_segments(
            X_list, y_list, weights_list
        )
        if verbose:
            n_zvf = zero_var_feats.sum()
            n_zvc = zero_var_chans.sum()
            if n_zvf or n_zvc:
                LOGGER.warning(
                    "Zero-variance: %d features, %d channels set to zeros.",
                    n_zvf, n_zvc,
                )

    # --- observed statistic ---
    # perm_norm uses zscore coef as its base statistic, then normalizes by null dispersion
    base_stat = "zscore" if stat == "perm_norm" else stat
    if n_segments == 1:
        X_obs, y_obs = X_list[0], y_list[0]
        w_obs = weights_list[0] if weights_list else None
    else:
        X_obs, y_obs = X_list, y_list
        w_obs = weights_list

    obs_stat = _fit_and_get_stat(
        trf, X_obs, y_obs, base_stat, weights=w_obs, lagged=lagged, zscored=zscored
    )

    # --- family mask ---
    family_mask = _resolve_family(family, obs_stat.shape)

    # --- observed max-stat ---
    obs_max = _max_stat(obs_stat, tails, family_mask)

    # --- permutation null ---
    rng = np.random.default_rng(seed)

    # Pre-compute valid offsets per segment
    valid_offsets_per_seg = [
        _valid_offsets(len(Xi), tmin_samp, tmax_samp) for Xi in X_list
    ]

    # Generate permutation seeds (deterministic, reproducible across n_jobs)
    perm_seeds = rng.integers(0, 2**63 - 1, size=n_perm)

    def _run_permutation(perm_idx):
        """Run a single permutation and return the null stat map."""
        local_rng = np.random.default_rng(perm_seeds[perm_idx])
        X_perm_list = []
        for i, Xi in enumerate(X_list):
            offsets = valid_offsets_per_seg[i]
            offset = int(local_rng.choice(offsets))
            X_perm_list.append(
                _circular_shift(Xi, offset, fade_samples=fade_per_seg[i])
            )

        if n_segments == 1:
            X_p, y_p = X_perm_list[0], y_list[0]
            w_p = weights_list[0] if weights_list else None
        else:
            X_p, y_p = X_perm_list, y_list
            w_p = weights_list

        null_stat = _fit_and_get_stat(
            trf, X_p, y_p, base_stat, weights=w_p, lagged=lagged, zscored=zscored
        )
        return null_stat

    if n_jobs == 1:
        null_maps = [_run_permutation(i) for i in range(n_perm)]
    else:
        try:
            from joblib import Parallel, delayed
            null_maps = list(
                Parallel(n_jobs=n_jobs)(
                    delayed(_run_permutation)(i) for i in range(n_perm)
                )
            )
        except ImportError:
            LOGGER.warning("joblib not available, running serially.")
            null_maps = [_run_permutation(i) for i in range(n_perm)]

    # --- compute p-values ---
    zero_var_stat = None

    if stat == "perm_norm":
        # Two-pass normalization: collect all maps (observed + nulls),
        # compute per-coordinate SD, normalize, then FWE on max|Z|.
        all_maps = np.array([obs_stat] + null_maps)  # (n_perm+1, n_lags, n_feats, n_chans)
        sd_j = np.std(all_maps, axis=0, ddof=1)
        eps = _EPS
        zero_var_stat = sd_j < eps
        sd_safe = np.where(zero_var_stat, 1.0, sd_j)
        Z = all_maps / sd_safe[None, ...]
        Z[:, zero_var_stat] = 0.0

        # FWE: max|Z| per map
        z_max = np.max(np.abs(Z.reshape(Z.shape[0], -1)), axis=1)
        null_max_stats = z_max[1:]  # null max-stats

        # Plus-one p-values from normalized ensemble
        obs_z = Z[0]
        pvals = _plus_one_pvals(obs_z, null_max_stats, tails, family_mask)
        obs_stat = obs_z  # report normalized observed stat
    else:
        # Standard max-stat FWE
        null_max_stats = np.array([
            _max_stat(nm, tails, family_mask) for nm in null_maps
        ])
        pvals = _plus_one_pvals(obs_stat, null_max_stats, tails, family_mask)

    hypothesis = (
        "H0: no stimulus→response predictive relationship at any modeled lag"
    )
    fade_str = (
        f"with edge fade ({fade_per_seg[0]} samples)" if fade_per_seg[0] > 0
        else "no edge fade"
    )
    resampling_scheme = f"circular-shift surrogate (per-segment, valid offsets, {fade_str})"

    return PermutationResult(
        observed=obs_stat,
        null_distribution=null_max_stats,
        pvals_corrected=pvals,
        stat=stat,
        tails=tails,
        family=family if isinstance(family, str) else "custom",
        n_perm=n_perm,
        seed=seed,
        hypothesis=hypothesis,
        resampling_scheme=resampling_scheme,
        zero_var_features=zero_var_feats,
        zero_var_channels=zero_var_chans,
        zero_var_stat=zero_var_stat,
    )


def _resolve_family(family, shape):
    """Resolve the family parameter to a boolean mask."""
    if isinstance(family, str):
        if family == "global":
            return np.ones(shape, dtype=bool)
        elif family == "per_channel":
            return np.ones(shape, dtype=bool)  # same as global for max-stat
        elif family == "per_feature":
            return np.ones(shape, dtype=bool)  # same as global for max-stat
        else:
            raise ValueError(f"Unknown family: {family!r}")
    elif isinstance(family, np.ndarray):
        if family.shape != shape:
            raise ValueError(
                f"family mask shape {family.shape} != stat map shape {shape}"
            )
        return family.astype(bool)
    else:
        raise TypeError(f"family must be str or ndarray, got {type(family)}")


# ===========================================================================
# Public API — jackknife SE estimator (standalone, not a permutation stat)
# ===========================================================================


def jackknife_se_trf(
    trf,
    X,
    y,
    alpha: float = 0.05,
    stat: str = "zscore",
    weights=None,
) -> JackknifeResult:
    """Leave-one-epoch-out jackknife standard error and confidence intervals.

    Estimates the sampling variability of TRF coefficients by leaving one
    epoch (segment) out at a time, refitting on the remaining N-1 epochs,
    and computing the jackknife standard error.

    This is a standalone uncertainty estimator, **not** a permutation test.
    The spike validation showed that using jackknife studentization as a
    permutation statistic has low power (the SE captures both signal and
    noise variability, shrinking the t-statistic).  Instead, the jackknife
    is offered as a lightweight SE/CI estimator alongside the bootstrap.

    Parameters
    ----------
    trf : TRFEstimator
        Configured TRF estimator (cloned and frozen for all refits).
    X : list of ndarray
        Multi-epoch stimulus features (list of segments, each
        (n_samples_i, n_feats)).  Single 2-D arrays are rejected —
        jackknife requires multiple epochs.
    y : list of ndarray
        Multi-epoch response (list of segments, each (n_samples_i, n_chans)).
    alpha : float, default 0.05
        CI confidence level (1 - alpha).  Uses normal approximation
        (z = 1.96 for 95%).
    stat : {"zscore", "coef"}
        If ``"zscore"``, z-score each epoch independently before fitting
        (scale-standardised coefficient).  If ``"coef"``, use raw coefficients.
    weights : list of ndarray or None
        Per-epoch sample weights (not supported for jackknife; raises).

    Returns
    -------
    result : JackknifeResult

    Notes
    -----
    Requires N >= 3 epochs.  Works with any fitted TRF configuration (OLS,
    ridge, banded ridge, robust) — the jackknife SE measures the sampling
    variability of the specific estimator, which is a valid uncertainty
    measure regardless of whether the estimator is biased (ridge) or
    iterative (robust).  For robust loss with few epochs the jackknife may
    be less stable.
    """
    from scipy.stats import norm

    # --- validate multi-epoch ---
    if isinstance(X, np.ndarray) and X.ndim == 2:
        raise ValueError(
            "jackknife_se_trf requires multi-epoch data (list of segments). "
            "A single 2-D array was provided."
        )

    X_list, y_list = _as_segment_list(X, y)
    N = len(X_list)
    if N < 3:
        raise ValueError(
            f"jackknife_se_trf requires N >= 3 epochs, got {N}."
        )

    if weights is not None:
        raise ValueError(
            "weights are not supported for jackknife_se_trf in v1."
        )

    # --- standardisation (if zscore) ---
    zscored = stat == "zscore"
    if stat not in ("zscore", "coef"):
        raise ValueError(f"stat must be 'zscore' or 'coef', got {stat!r}.")

    if zscored:
        X_list, y_list, _, _ = _zscore_segments(X_list, y_list)

    # --- full-data fit ---
    clone_full = _clone_trf(trf, fit_intercept=False if zscored else None)
    clone_full.fit(X_list, y_list)
    coef_full = clone_full.coef_.copy()

    # --- leave-one-epoch-out ---
    thetas_loo = []
    for i in range(N):
        X_loo = [X_list[j] for j in range(N) if j != i]
        y_loo = [y_list[j] for j in range(N) if j != i]
        clone_loo = _clone_trf(trf, fit_intercept=False if zscored else None)
        clone_loo.fit(X_loo, y_loo)
        thetas_loo.append(clone_loo.coef_.copy())

    thetas_loo = np.array(thetas_loo)  # (N, n_lags, n_feats, n_chans)
    theta_bar = thetas_loo.mean(axis=0)

    # Jackknife SE
    se = np.sqrt((N - 1) / N * np.sum((thetas_loo - theta_bar[None, ...]) ** 2, axis=0))

    # Handle zero SE
    eps = _EPS
    se_safe = np.where(se < eps, np.inf, se)

    # Normal-approximation CI
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = coef_full - z_crit * se_safe
    ci_high = coef_full + z_crit * se_safe
    # Where SE is zero, CI = coef (no variability)
    ci_low[se < eps] = coef_full[se < eps]
    ci_high[se < eps] = coef_full[se < eps]

    return JackknifeResult(
        coef=coef_full,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_epochs=N,
        alpha=alpha,
    )


# ===========================================================================
# Public API — stubs (to be implemented)
# ===========================================================================


def cluster_based_permutation_test(
    trf,
    X,
    y,
    n_perm: int = 1000,
    threshold=None,
    adjacency="lags",
    family="global",
    tails="two-sided",
    stat="zscore",
    n_jobs: int = 1,
    seed=None,
    allow_robust: bool = False,
    weights=None,
    lagged: bool = False,
) -> ClusterResult:
    """Cluster-based permutation test (Maris & Oostenveld 2007).

    Not yet implemented — see plan v6 §2.2.
    """
    raise NotImplementedError("cluster_based_permutation_test not yet implemented.")


def bootstrap_ci_trf(
    trf,
    X,
    y,
    n_boot: int = 1000,
    block_size="auto",
    method="circular",
    interval="percentile",
    alpha=0.05,
    stat="zscore",
    n_jobs: int = 1,
    seed=None,
    return_distribution: bool = False,
    weights=None,
    lagged: bool = False,
) -> BootstrapResult:
    """Paired block-bootstrap confidence intervals on TRF coefficients.

    Not yet implemented — see plan v6 §2.3.
    """
    raise NotImplementedError("bootstrap_ci_trf not yet implemented.")


def cross_subject_consistency(
    trfs,
    metric="corr",
    leave_one_out: bool = False,
) -> ConsistencyResult:
    """Descriptive cross-subject consistency of TRF coefficients.

    Not yet implemented — see plan v6 §2.4.
    """
    raise NotImplementedError("cross_subject_consistency not yet implemented.")


def group_level_test(
    trfs,
    n_perm: int = 1000,
    family="global",
    tails="two-sided",
    n_jobs: int = 1,
    seed=None,
) -> GroupTestResult:
    """Group-level sign-flip test on subject coefficient maps.

    Not yet implemented — see plan v6 §2.4.
    """
    raise NotImplementedError("group_level_test not yet implemented.")


class TRFAnalyzer:
    """Convenience facade for TRF statistical analysis.

    Wraps a (fitted) TRFEstimator + its X, y and chains the stats methods.

    Not yet implemented — see plan v6 §6.
    """

    def __init__(self, trf, X=None, y=None):
        self.trf = trf
        self.X = X
        self.y = y

    def permutation_test(self, **kw):
        return permutation_test_trf(self.trf, self.X, self.y, **kw)

    def cluster_based_test(self, **kw):
        return cluster_based_permutation_test(self.trf, self.X, self.y, **kw)

    def bootstrap_ci(self, **kw):
        return bootstrap_ci_trf(self.trf, self.X, self.y, **kw)

    @staticmethod
    def cross_subject(trfs, **kw):
        return cross_subject_consistency(trfs, **kw)

    @staticmethod
    def group_test(trfs, **kw):
        return group_level_test(trfs, **kw)
