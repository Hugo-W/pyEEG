# pylint: disable=invalid-name,wrong-import-position,unsubscriptable-object
"""
In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

* Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
* Backward modelling (EEG -> stimulus)
* CCA (in :mod:pyeeg.cca)
* VAR model fitting

Updates:
- 10/11/2023: added VAR model estimation (see :func:`fit_var` and :func:`fit_ar`)

"""

import logging

# logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from .solvers import _robust_irls_regress, _robust_least_squares_regress, _svd_regress
from .utils import design_lagmatrix, lag_matrix, lag_span, lag_sparse, mem_check
from .vizu import get_spatial_colors, plot_interactive

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__.split(".")[0])


def fit_ar(x, nlags=1, time_axis=0):
    """
    Fit an autoregressive model to a time series.
    This is a helper function for autoregressive models estimation.

    If fed with a multidimensional time series, it will fit a model for each
    dimension.

    Parameters
    ----------
    x : ndarray (nsamples, nchans) or (nsamples, )
        Time series to fit.
    nlags : int
        Number of lags to use in the model (model order).
    time_axis : int
        Axis of the time series.

    Returns
    -------
    betas : ndarray (nchans, nlags)
        Coefficients of the autoregressive model.
    """
    if x.ndim == 1:
        time_axis = 1
    x = np.atleast_2d(x)
    if time_axis == 1:  # transpose if time is in columns
        x = x.T
    n, k = x.shape  # n: number of observations, k: number of dimensions

    X = design_lagmatrix(
        x, nlags=nlags, time_axis=0
    )  # time axis was already transposed
    if k == 1:
        X = np.atleast_3d(X)
    Y = x[nlags:, :]

    betas = np.zeros((k, nlags))
    for i in range(k):
        betas[i] = np.linalg.lstsq(X[:, :, i], Y[:, i], rcond=None)[0]
    return betas.squeeze() if k == 1 else betas


def fit_var(x, nlags=1, time_axis=0):
    """
    Fit a VAR model to a time series.

    Instead of fitting k independent models as in `fit_ar`, this function fits a single model
    but with multivariate regressors.

    Parameters
    ----------
    x : ndarray (nsamples, nchans) or (nsamples, )
        Time series to fit.
    nlags : int
        Number of lags to use in the model (model order).
    time_axis : int
        Axis of the time series.

    Returns
    -------
    betas : ndarray (nchans, nlags)
        Coefficients of the autoregressive model.
    """
    if x.ndim == 1:
        time_axis = 1
    x = np.atleast_2d(x)
    if time_axis == 1:  # transpose if time is in columns
        x = x.T
    n, k = x.shape  # n: number of observations, k: number of dimensions

    X = design_lagmatrix(
        x, nlags=nlags, time_axis=0
    )  # time axis was already transposed
    if k == 1:
        X = np.atleast_3d(X)
    Y = x[nlags:, :]
    # Now instead of looping over thrid axies of X (dimensions), we reshape it to a 2D matrix
    # And fit a single model
    betas = np.linalg.lstsq(X.reshape(-1, nlags * k), Y, rcond=None)[0]
    return betas.reshape(k, nlags, k)  # .transpose(2, 1, 0) # reshape back to 3D


class TRFEstimator(BaseEstimator):
    """Temporal Response Function (TRF) Estimator Class.

    This class allows to estimate TRF from a set of feature signals and an EEG dataset in the same fashion
    than :class:`mne.decoding.ReceptiveFieldEstimator` does in **MNE**.
    However, an arbitrary set of lags can be given. Namely, it can be used in two ways

    - calling with ``tmin`` and ``tmax`` arguments will compute lags spanning from ``tmin`` to ``tmax``.
    - with the ``times`` argument, one can request an arbitrary set of time lags at which to compute \
    the coefficients of the TRF.


    Attributes
    ----------
    lags : 1d-array
        Array of ``int``, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of ``float``, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    use_regularisation : bool
        Whether or not regularisation is used to compute the TRF
    feature_alphas : 1d-array or None
        Optional per-feature ridge strengths. Each value is repeated over all
        lags, following ``block_order``.
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    fitted : bool
        True once the TRF has been fitted on EEG data
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    tvals_ : ndarray (nlags * nfeats, nchans)
        t-statistics in the flattened canonical ordering of ``coef_``.
    pvals_ : ndarray (nlags * nfeats, nchans)
        Two-sided p-values in the flattened canonical ordering of ``coef_``.
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features

    Notes
    -----
    * Attributes with a ``_`` suffix are only set once the TRF has been fitted on EEG data (i.e. after \
    the method :meth:`TRFEstimator.fit` has been called).
    * Can fit on a list of multiple dataset, where we have a list of target ``Y`` and \
    a single stimulus matrix of features ``X``, then the computation is made such that \
    the coefficients computed are similar to those obtained by concatenating all matrices
   

    Examples
    --------
    >>> trf = TRFEstimator(tmin=-0.5, tmax-1.2, srate=125)
    >>> x = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000, 2)
    >>> trf.fit(x, y, lagged=False)

    .. seealso::
        :func:`_svd_regress`
    """

    def fromArray(arr, tmin, tmax, fs):
        """
        Creates a TRF instance from a 3D array.

        Parameters
        ----------
        arr : ndarray (nlags, nfeats, nchans)
            TRF weights.
        tmin : float
            Minimum lag in sec.
        tmax : float
            Maximum lag in sec.
        fs : float
            Sampling rate.

        Returns
        -------
        trf : TRFEstimator instance
        """
        trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=fs)
        trf.fill_lags()
        assert arr.shape[0] == len(trf.lags), (
            "Mismatch in lags! Supplied array has %d in first dimension, while %d lags were spanned"
            % (arr.shape[0], len(trf.lags))
        )
        trf.coef_ = arr
        trf.n_chans_ = arr.shape[-1]
        trf.n_feats_ = arr.shape[1]
        trf.fitted = True
        return trf

    def __init__(
        self,
        times=(0.0,),
        tmin=None,
        tmax=None,
        srate=1.0,
        alpha=None,
        fit_intercept=True,
        verbose=True,
        quadratic_reg=None,
        block_order="lags",
        loss="linear",
        robust_solver="irls",
        robust_sigma=None,
        robust_max_iter=20,
        robust_tol=1e-6,
        robust_damping=1.0,
        robust_inner_solver="svd",
        robust_inner_tol=1e-8,
        robust_inner_max_iter=None,
        feature_alphas=None,
    ):

        if block_order not in ("lags", "features"):
            raise ValueError("block_order must be 'lags' or 'features'")
        if loss not in ("linear", "none", "cauchy"):
            raise ValueError("loss must be 'linear', 'none', or 'cauchy'")
        if robust_solver not in ("irls", "least_squares"):
            raise ValueError("robust_solver must be 'irls' or 'least_squares'")
        if robust_inner_solver not in ("svd", "cg"):
            raise ValueError("robust_inner_solver must be 'svd' or 'cg'")
        if robust_sigma is not None and (
            not np.isscalar(robust_sigma)
            or robust_sigma <= 0
            or not np.isfinite(robust_sigma)
        ):
            raise ValueError("robust_sigma must be positive and finite")
        if feature_alphas is not None:
            feature_alphas = np.asarray(feature_alphas, dtype=float)
            if feature_alphas.ndim != 1 or feature_alphas.size == 0:
                raise ValueError("feature_alphas must be a non-empty 1-D array")
            if np.any(~np.isfinite(feature_alphas)) or np.any(feature_alphas < 0):
                raise ValueError(
                    "feature_alphas must contain finite non-negative values"
                )
            if quadratic_reg is not None:
                raise ValueError("feature_alphas cannot be combined with quadratic_reg")
            if alpha is not None and np.ndim(alpha) != 0:
                raise ValueError("feature_alphas cannot be combined with an alpha path")
        self.feature_alphas = feature_alphas
        if robust_max_iter < 1 or robust_tol <= 0 or not 0 < robust_damping <= 1:
            raise ValueError("Invalid robust iteration parameters")
        self.block_order = block_order
        self.loss = "linear" if loss == "none" else loss
        self.robust_solver = robust_solver
        self.robust_sigma = robust_sigma
        self.robust_max_iter = robust_max_iter
        self.robust_tol = robust_tol
        self.robust_damping = robust_damping
        self.robust_inner_solver = robust_inner_solver
        self.robust_inner_tol = robust_inner_tol
        self.robust_inner_max_iter = robust_inner_max_iter
        # if tmin and tmax:
        #     LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
        #     self.lags = lag_span(tmin, tmax, srate=srate)[::-1] #pylint: disable=invalid-unary-operand-type
        #     #self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
        #     self.times = self.lags[::-1] / srate
        # else:
        #     self.times = np.asarray(times)
        #     self.lags = lag_sparse(self.times, srate)[::-1]

        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        # Quadratic regularization: str ('smoothness'/'laplacian') or ndarray.
        # M REPLACES the L2 regularization; alpha becomes the M-strength knob.
        self.quadratic_reg = quadratic_reg
        # alpha is the regularization strength. When quadratic regularization (M)
        # is active, alpha scales M and defaults to 1 if not provided; without M,
        # alpha=None means unregularized (plain least squares).
        if self.quadratic_reg is not None and alpha is None:
            alpha = 1.0
        elif alpha is None:
            alpha = 0.0
        self.alpha = alpha
        self.verbose = verbose
        # Regularization is on when alpha > 0 OR when quadratic regularization (M)
        # is requested (M counts as regularization even when alpha == 0, in which
        # case M is scaled to zero and the fit reduces to plain least squares)
        if np.ndim(self.alpha) == 0:
            self.use_regularisation = (
                self.alpha > 0.0
                or self.quadratic_reg is not None
                or self.feature_alphas is not None
            )
        else:
            self.use_regularisation = (
                np.any(np.asarray(self.alpha) > 0.0)
                or self.quadratic_reg is not None
                or self.feature_alphas is not None
            )
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.rotations_ = None  # matrices to be used to rotate coefficients into a 'better conditonned subspace'
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # The two following are only defined if simple least-square (no reg.) is used
        self.tvals_ = None
        self.pvals_ = None
        self.robust_n_iter_ = None
        self.robust_converged_ = None
        self.robust_scale_ = None
        self.robust_objective_ = None

    def fill_lags(self):
        """Fill the lags attributes.

        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.

        """
        if (self.tmin is not None) and (self.tmax is not None):
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[::-1]  # pylint: disable=invalid-unary-operand-type
            # self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]

    def _beta_to_coef(self, betas):
        """Map a flattened (n_lags*n_feats, n_chans) solver beta block to the
        public ``coef_`` shape (n_lags, n_feats, n_chans).

        The solver beta columns follow ``lag_matrix(..., block_order=...)``:
        - ``'lags'``: [feat0_lag0, feat1_lag0, feat0_lag1, feat1_lag1, ...]
        - ``'features'``: [feat0_lag0, feat0_lag1, ..., feat1_lag0, ...]

        ``coef_`` is always stored with shape ``(n_lags, n_feats, n_chans)``;
        its first axis follows the estimator's public ``times`` ordering. The
        lag axis is flipped relative to the solve order (``lag_matrix``
        negates the lags internally), preserving the legacy convention.
        """
        betas = np.asarray(betas)
        n_lags = len(self.lags)
        n_feats = self.n_feats_
        if self.block_order == "lags":
            coef = betas.reshape(n_lags, n_feats, -1)
        else:  # 'features': each feature's lags are contiguous in the beta block
            coef = betas.reshape(n_feats, n_lags, -1).swapaxes(0, 1)
        return coef[::-1, :, :]  # flip lag axis to increasing lag order

    def _coef_to_beta(self, coef):
        """Inverse of :meth:`_beta_to_coef`: map a (n_lags, n_feats, n_chans)
        ``coef_`` back to the flattened solver beta column order for the
        estimator's ``block_order`` (used by prediction-related paths)."""
        coef = np.asarray(coef)
        n_lags = len(self.lags)
        n_feats = self.n_feats_
        coef_flipped = coef[::-1, :, :]  # back to solve lag order
        if self.block_order == "lags":
            return coef_flipped.reshape(n_lags * n_feats, -1)
        return coef_flipped.swapaxes(0, 1).reshape(n_lags * n_feats, -1)

    def _build_quadratic_regularizer(self):
        """Build the quadratic regularization matrix M from ``self.quadratic_reg``.

        M REPLACES the L2 (``alpha``) regularization: ``alpha`` acts as the
        M-strength knob (``alpha=0`` scales M to zero, reducing the fit to plain
        least squares). When ``alpha`` is not provided (``None``) it defaults to 1
        in :meth:`__init__`.

        ``quadratic_reg`` accepts:
        - a string (``'smoothness'`` / ``'laplacian'``): M built and scaled by
          ``alpha`` via :func:`pyeeg.solvers.create_quadratic_regularizer`;
        - an ndarray: used as-is, scaled by ``alpha``. Custom matrices are
          currently expected to already match the full solver design order and
          shape. With ``fit_intercept=True``, provide a zero-padded leading
          intercept row and column; feature/lag ordering must match
          ``block_order``. Automatic predictor-space custom-matrix padding is
          intentionally deferred.

        Returns
        -------
        M : ndarray or None
            Regularization matrix in solver column order, or None when no
            quadratic regularization is requested. ``feature_alphas`` creates
            a diagonal banded-ridge matrix; ``quadratic_reg`` creates the
            configured smoothness/custom matrix.
        """
        if self.feature_alphas is not None:
            if self.n_feats_ is None or len(self.lags) == 0:
                raise RuntimeError("feature_alphas requires a fitted lag configuration")
            if len(self.feature_alphas) != self.n_feats_:
                raise ValueError(
                    "feature_alphas must have one value per input feature "
                    f"({self.n_feats_} expected, got {len(self.feature_alphas)})"
                )
            if self.block_order == "lags":
                diagonal = np.tile(self.feature_alphas, len(self.lags))
            else:
                diagonal = np.repeat(self.feature_alphas, len(self.lags))
            M = np.diag(diagonal)
            if self.fit_intercept:
                M = np.pad(M, ((1, 0), (1, 0)))
            return M
        if self.quadratic_reg is None:
            return None
        if isinstance(self.quadratic_reg, str):
            from pyeeg.solvers import create_quadratic_regularizer

            n_lags = len(self.lags)
            n_feats = self.n_feats_
            # alpha is the M-strength knob (scales the per-feature Laplacian)
            L_single = create_quadratic_regularizer(
                self.quadratic_reg, n_lags, alpha=self.alpha
            )
            if self.block_order == "lags":
                # Lag-major column ordering:
                # [feat0_lag0, feat1_lag0, feat0_lag1, feat1_lag1, ...]
                # Temporal smoothness couples equal-feature entries across
                # lag blocks: L ⊗ I_features.
                M = np.kron(L_single, np.eye(n_feats))
            else:
                # Feature-major column ordering:
                # [feat0_lag0, feat0_lag1, ..., feat1_lag0, feat1_lag1, ...]
                # Each feature occupies one contiguous L-sized block.
                M = np.kron(np.eye(n_feats), L_single)
            # When an intercept column is present the solver beta block has
            # n_lags*n_feats + 1 entries; pad M so the intercept stays
            # unregularized (all-zero row/column).
            if self.fit_intercept:
                M = np.pad(M, ((1, 0), (1, 0)))
            return M
        # Pre-built matrix: scale by alpha (alpha=0 -> M vanishes -> least squares)
        return np.asarray(self.quadratic_reg) * self.alpha

    def _fit_robust(self, X, y, M):
        """Fit the configured Cauchy-loss model and store convergence metadata."""
        if self.robust_solver == "least_squares":
            if self.use_regularisation:
                raise ValueError(
                    "robust_solver='least_squares' currently supports only "
                    "unregularized fitting (alpha=0, feature_alphas=None, "
                    "and quadratic_reg=None)."
                )
            betas, info = _robust_least_squares_regress(
                X, y, scale=self.robust_sigma, verbose=self.verbose
            )
        else:
            betas, info = _robust_irls_regress(
                X,
                y,
                alpha=self.alpha,
                M=M,
                loss="cauchy",
                scale=self.robust_sigma,
                max_iter=self.robust_max_iter,
                tol=self.robust_tol,
                damping=self.robust_damping,
                inner_solver=self.robust_inner_solver,
                inner_tol=self.robust_inner_tol,
                inner_max_iter=self.robust_inner_max_iter,
                verbose=self.verbose,
            )
        self.robust_n_iter_ = info["n_iter"]
        self.robust_converged_ = info["converged"]
        self.robust_scale_ = info["scale"]
        self.robust_objective_ = info.get("objective")
        return betas

    def fit(
        self, X, y, lagged=False, drop=True, feat_names=(), rotations=(), weights=None
    ):
        """Fit the TRF model.

        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged or not, if it is, then second dim's shape should be nfeats*nlags)
        y : ndarray (nsamples x nchans)
            EEG data
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        rotations : list of ndarrays (shape (nlag x nlags))
            List of rotation matrices (if ``V`` is one such rotation, ``V @ V.T`` is a projection).
            Can use empty item in place of identity matrix.
        weights : ndarray (nsamples,), optional
            Sample weights for weighted least squares. If provided, each sample
            is scaled by ``sqrt(weights)`` (after dropping invalid samples and
            lagging, before the intercept is added). Must be non-negative and
            of length ``n_samples`` (the full length, before dropping). Sample
            weights are not currently combined with robust fitting.

        Notes
        -----
        Set ``loss='cauchy'`` in the constructor to fit the Cauchy loss
        ``log(1 + (residual / robust_sigma)**2)``. The default ``loss='linear'``
        preserves the ordinary least-squares and ridge paths. Robust fitting
        uses IRLS by default; ``robust_solver='least_squares'`` selects SciPy's
        nonlinear Cauchy solver for small, unregularized dense problems.
        Classical t- and p-values are not computed for robust fits.

        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans)
        intercept_ : ndarray (nchans x 1)
        """
        self.fill_lags()

        if isinstance(y, list) and isinstance(X, list):
            if self.verbose:
                LOGGER.info(
                    "Supplied a list of data portions... Will compute covariance matrices by 'accumulating' them."
                )
            assert len(y) == len(X), (
                "Both lists (X and y) should have the same number of elements"
            )
            assert all([len(yy) == len(xx) for xx, yy in zip(X, y)]), (
                "Each data portion should have the same number of samples"
            )
            return self._fitlists(
                X, y, drop, feat_names, lagged, self.verbose, weights=weights
            )

        y = np.asarray(y)
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(y) == 3 else y.nbytes
        estimated_mem_usage = (
            sum([x.nbytes for x in X]) if np.ndim(X) == 3 else X.nbytes
        ) * (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage / 1024.0**3 > mem_check():
            raise MemoryError(
                "Not enough RAM available! (needed %.1fGB, but only %.1fGB available)"
                % (estimated_mem_usage / 1024.0**3, mem_check())
            )

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        n_samples_all = (
            y.shape[0] if y.ndim == 2 else y.shape[1]
        )  # this include non-valid samples for now

        if drop:
            self.valid_samples_ = np.logical_not(
                np.logical_or(
                    np.arange(n_samples_all) < abs(max(self.lags)),
                    np.arange(n_samples_all)[::-1] < abs(min(self.lags)),
                )
            )
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        if self.verbose:
            LOGGER.info("Lagging matrix...")
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:, self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(
                X,
                lags=self.lags,
                mode="valid" if drop else "full",
                fill_value=np.nan if drop else 0.0,
                block_order=self.block_order,
            )
        elif len(X) == n_samples_all:
            # Pre-lagged callers may provide the full sample axis. Align it
            # with y when edge samples are dropped; already-trimmed inputs are
            # left unchanged.
            X = X[self.valid_samples_, :]
        """
        if not lagged:
            if drop:
                X = lag_matrix(X, lags=self.lags, mode='valid')

                # Droping rows of NaN values in y
                if any(np.asarray(self.lags) < 0):
                    drop_top = abs(min(self.lags))
                    y = y[drop_top:, :] if y.ndim == 2 else y[:, drop_top:, :]
                if any(np.asarray(self.lags) > 0):
                    drop_bottom = abs(max(self.lags))
                    y = y[:-drop_bottom, :] if y.ndim == 2 else y[:, :-drop_bottom, :]
            else:
                X = lag_matrix(X, lags=self.lags, mode='full', fill_value=0.)
        """
        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        robust = self.loss == "cauchy"
        if robust and weights is not None:
            raise ValueError("weights cannot currently be combined with loss='cauchy'.")

        # Apply sample weights (weighted least squares) after the intercept is
        # added: every column of X (including the intercept) is row-scaled by
        # sqrt(weights), so the existing solvers solve the weighted problem
        # unchanged. Weights are indexed against the full sample axis, so slice
        # them with valid_samples_ to align with the post-drop X/y rows.
        if weights is not None:
            from pyeeg.utils import apply_sample_weights

            w = np.asarray(weights, dtype=float)[self.valid_samples_]
            if np.any(w < 0):
                raise ValueError("Sample weights must be non-negative.")
            X, y = apply_sample_weights(X, y, w)
            if self.verbose:
                LOGGER.info("Applied sample weights (weighted least squares)")

        # Solving with robust IRLS/SciPy or the existing linear solvers.
        if self.verbose:
            LOGGER.info("Computing coefficients..")
        M = self._build_quadratic_regularizer()
        if robust:
            betas = self._fit_robust(X, y, M)
            self.all_betas = betas[..., np.newaxis]
        elif self.use_regularisation or np.ndim(y) == 3:
            betas = _svd_regress(X, y, self.alpha, M=M, verbose=self.verbose)
            self.all_betas = betas
            # Storing only the first as the main
            betas = betas[..., 0]
        else:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        if rotations:
            if self.block_order != "lags":
                raise NotImplementedError(
                    "rotations are only supported with block_order='lags' "
                    "(the legacy lag-major layout); block_order='features' "
                    "rotations are not implemented."
                )
            newbetas = np.zeros((len(self.lags) * self.n_feats_, self.n_chans_))
            for _, rot in zip(range(self.n_feats_), rotations):
                if not rot:
                    rot = np.eye(self.lags)
                newbetas[:: self.n_feats_, :] = rot @ betas[...]
            betas = newbetas

        self.coef_ = self._beta_to_coef(betas)
        self.fitted = True

        # Compute standardized coefficients (beta * std(X) / std(y))
        # This gives the change in y (in SDs) for a 1 SD change in X.
        # Skipped when sample weights are applied: X/y have been row-scaled by
        # sqrt(W) here, so their std no longer reflects the original data.
        if not lagged and self.fitted and weights is None and not robust:
            try:
                X_std = np.std(
                    X[:, self.fit_intercept :], axis=0, keepdims=False
                )  # (n_lags*n_feats,)
                y_std = np.std(y, axis=0, keepdims=True)  # (1, n_chans)
                # Map the per-column stds (which follow the estimator's
                # block_order) to coef_ shape: (n_lags, n_feats, 1)
                X_std_reshaped = self._beta_to_coef(X_std[:, None])
                self.standardized_coef_ = (
                    self.coef_ * X_std_reshaped / y_std[None, None, :]
                )
            except Exception as e:
                if self.verbose:
                    LOGGER.warning("Could not compute standardized coefficients: %s", e)
                self.standardized_coef_ = None
        else:
            self.standardized_coef_ = None

        # Get t-statistic and p-vals if regularization is ommited
        # (M counts as regularization: t-values computed from X.T@X would ignore it)
        if not self.use_regularisation and not robust:
            if self.verbose:
                LOGGER.info("Computing statistics...")
            # Intercept column is present in X / cov_betas only when fit_intercept
            n_intercept = int(self.fit_intercept)
            cov_betas = X.T @ X
            # Compute variance sigma (MSE)
            if np.ndim(y) == 3:
                dof = sum(list(map(len, y))) - (len(betas) + n_intercept)
                sigma = 0.0
                for yy in y:
                    sigma += np.sum((yy - self.predict(X)) ** 2, axis=0)
                sigma /= dof
            else:
                dof = len(y) - (len(betas) + n_intercept)
                sigma = np.sum((y - self.predict(X)) ** 2, axis=0) / dof
            # Covariance matrix on betas
            try:
                cov_betas_inv = np.linalg.inv(cov_betas)
            except np.linalg.LinAlgError:
                # Rank-deficient design matrix: fall back to the pseudo-inverse
                cov_betas_inv = np.linalg.pinv(cov_betas)
            C = np.einsum("ij,k", cov_betas_inv, sigma)
            # Actual stats (strip the intercept entry of the diagonal iff present)
            se = np.sqrt(C.diagonal(axis1=0, axis2=1).swapaxes(0, 1)[n_intercept:, :])
            # tvals_/pvals_ are stored in the same canonical flattened ordering
            # as coef_ (issue #30): map the solver-order betas/se through
            # _beta_to_coef (after intercept removal) so each coefficient
            # aligns 1:1 with coef_. The solver-order SE computation is
            # preserved; only the public ordering of the outputs changes.
            tvals = self._beta_to_coef(betas / se)
            self.tvals_ = tvals.reshape(-1, tvals.shape[-1])
            # Use the survival function (sf = 1 - cdf) instead of "1 - cdf"
            # to avoid catastrophic cancellation: for large |t| the cdf
            # saturates to 1.0 in float64 and "1 - 1.0" underflows to 0.0,
            # yielding spurious exact-zero p-values. sf is computed
            # accurately in the far tail.
            self.pvals_ = 2 * stats.t.sf(abs(self.tvals_), df=dof)

        return self

    def _fitlists(
        self, X, y, drop=True, feat_names=(), lagged=False, verbose=True, weights=None
    ):
        """
        Fit the TRF by accumulating the covariance matrices of each item in the
        list of arrays in ``X`` and ``Y``.
        This is more memory efficient and can follow nicely an experiment design
        where several audio clips of variable length are aligned with M/EEG.

        Parameters
        ----------
        X : list of ndarray of shape (nsamples, nfeats) or (nsamples, nfeats*nlags)
            List of predictor data
        y : list of ndarray, shape (nsamples, nchans)
            List of M/EEG data
        drop : bool, optional
            Whether to drop invalid samples on lagged matrices. The default is True.
        feat_names : tuple (str), optional
            Feature names. The default is ().
        lagged : Bool, optional
            Whether the predictor matrices have been lagged already. The default is False.
        verbose : bool, optional
            The default is True.
        weights : list of ndarray, optional
            Sample weights for weighted least squares, one array per segment
            (same length as each ``y`` segment). If provided, each segment's X/y
            rows are scaled by ``sqrt(weights)`` after dropping invalid samples
            and lagging, before the intercept is added.

        Returns
        -------
        TRFEstimator
            Fitted instance of TRF model.

        """
        # if drop:
        # raise NotImplementedError("Please use drop=False, this feature has not been implemented yet")

        # For each element (subject or segment) in Y list, check which sample to drop
        valid_samples = []
        for yy in y:
            n_samples_all = yy.shape[0]
            if drop:
                valid_samples.append(
                    np.logical_not(
                        np.logical_or(
                            np.arange(n_samples_all) < abs(max(self.lags)),
                            np.arange(n_samples_all)[::-1] < abs(min(self.lags)),
                        )
                    )
                )
            else:
                valid_samples.append(np.ones((n_samples_all,), dtype=bool))

        self.n_chans_ = y[0].shape[1]
        self.n_feats_ = X[0].shape[1] if not lagged else X[0].shape[1] // len(self.lags)
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        # Build quadratic regularization matrix M if specified
        # (M replaces L2 regularization; alpha has no effect when M is active)
        M = self._build_quadratic_regularizer()
        robust = self.loss == "cauchy"
        if robust and weights is not None:
            raise ValueError("weights cannot currently be combined with loss='cauchy'.")

        if weights is not None:
            from pyeeg.utils import apply_sample_weights

            if len(weights) != len(X):
                raise ValueError("weights must be a list with one array per segment.")

        if lagged:
            X_list = (
                [np.hstack([np.ones((len(x), 1)), x]) for x in X]
                if self.fit_intercept
                else list(X)
            )
            y_list = [yy[s] for s, yy in zip(valid_samples, y)]
            if weights is not None:
                X_list_new, y_list_new = [], []
                for i, (xx, yy, w) in enumerate(zip(X_list, y_list, weights)):
                    wv = np.asarray(w, dtype=float)[valid_samples[i]]
                    if np.any(wv < 0):
                        raise ValueError("Sample weights must be non-negative.")
                    xw, yw = apply_sample_weights(xx, yy, wv)
                    X_list_new.append(xw)
                    y_list_new.append(yw)
                X_list, y_list = X_list_new, y_list_new
            if robust:
                betas = self._fit_robust(X_list, y_list, M)
            else:
                betas = _svd_regress(
                    X_list, y_list, self.alpha, M=M, verbose=self.verbose
                )
        else:
            filling = np.nan if drop else 0.0
            X_list = []
            y_list = []
            for i, (s, x, yy) in enumerate(zip(valid_samples, X, y)):
                xx = lag_matrix(
                    x,
                    self.lags,
                    fill_value=filling,
                    mode="valid" if drop else "full",
                    block_order=self.block_order,
                )
                yc = yy[s]
                if self.fit_intercept:
                    xx = np.hstack([np.ones((len(xx), 1)), xx])
                # The intercept column is added BEFORE sqrt-weighting so it is
                # weighted like every other column (correct WLS semantics,
                # consistent with the single-array path in :meth:`fit`).
                if weights is not None:
                    wv = np.asarray(weights[i], dtype=float)[s]
                    if np.any(wv < 0):
                        raise ValueError("Sample weights must be non-negative.")
                    xx, yc = apply_sample_weights(xx, yc, wv)
                X_list.append(xx)
                y_list.append(yc)
            if robust:
                betas = self._fit_robust(X_list, y_list, M)
            else:
                betas = _svd_regress(
                    X_list, y_list, self.alpha, M=M, verbose=self.verbose
                )
        # Preserve the existing alpha-path shape for linear fits. Robust
        # fitting has one coefficient solution per output channel.
        if robust:
            self.all_betas = betas[..., np.newaxis]
        else:
            self.all_betas = betas
            # Storing only the first as the main
            betas = betas[..., 0]

        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        self.coef_ = self._beta_to_coef(betas)

        self.fitted = True
        return self

    def select_best_coefs(self, best_index, in_place=False):
        """
        This method can be used to select the best set of coefficients when the
        TRF model has been trained with several regularisation parmaters.

        Parameters
        ----------
        best_index : int
            Index of best model (w.r.t alpha array/list).
        in_place : bool
            Whether to operate in-place (default to False).

        Returns
        -------
        :class:`TRFEstimator` instance
        """
        assert hasattr(self, "all_betas"), (
            "TRF must be fitted with several regularisation values alpha at once."
        )
        trf = self if in_place else self.copy()
        betas = self.all_betas[..., best_index]
        trf.alpha = self.alpha[best_index]
        if trf.fit_intercept:
            trf.intercept_ = betas[0, :]
            betas = betas[1:, :]
        trf.coef_ = trf._beta_to_coef(betas)
        return trf

    def plot_multialpha_scores(self, X, y):
        """
        Plot the score against different alphas to visualise effect of
        regularisation.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert hasattr(self, "all_betas"), (
            "TRF must be fitted with several regularisation values alpha at once."
        )
        scores = self.multialpha_score(X, y)
        scores_toplot = scores.mean(0).mean(-1).T
        # Best alpha
        peaks = scores.mean(0).mean(-1).argmax(1)
        lines = plt.semilogx(self.alpha, scores_toplot)
        if y.ndim == 3:  # multi-subject (search best alpha PER subject)
            for k, p in enumerate(peaks):
                plt.semilogx(
                    self.alpha[p],
                    scores_toplot[p, k],
                    "*",
                    ms=10,
                    color=lines[k].get_color(),
                )
        else:
            plt.semilogx(
                self.alpha[scores.mean(0).mean(-1).argmax()],
                scores_toplot[scores.mean(0).mean(-1).argmax()],
                "*k",
                ms=10,
            )

    def multialpha_score(self, X, y):
        assert hasattr(self, "all_betas"), (
            "TRF must be fitted with several regularisation values alpha at once."
        )
        # For several story-parts
        if isinstance(X, list) and len(X) == len(y):
            scores = np.mean(
                [
                    self.multialpha_score(x, yy)
                    for x, yy in tqdm(
                        zip(X, y), total=len(X), desc="Scoring each segment "
                    )
                ],
                0,
            )
            return scores
        else:
            # Lag X if necessary, and add intercept
            if X.shape[1] != (len(self.lags) * self.n_feats_ + int(self.fit_intercept)):
                X = lag_matrix(
                    X,
                    lags=self.lags,
                    mode="full",
                    fill_value=0.0,
                    block_order=self.block_order,
                )
                if self.fit_intercept:
                    X = np.hstack([np.ones((len(X), 1)), X])

            # Estimate yhat
            yhat = np.einsum("ij,jkl->ikl", X, self.all_betas)
            y = np.asarray(y)

            # Compute scores
            # A single X and a single y
            if y.ndim == 2:  # single-subject
                scores = np.zeros((1, len(self.alpha), self.n_chans_), dtype=y.dtype)
                for lamb in range(len(self.alpha)):
                    scores[0, lamb, :] = np.diag(
                        np.corrcoef(yhat[..., lamb], y, rowvar=False), k=self.n_chans_
                    )
            else:  # multi-subject (one X several ys)
                scores = np.zeros(
                    (y.shape[0], len(self.alpha), self.n_chans_), dtype=y[0].dtype
                )
                for ksubj, yy in enumerate(y):
                    for lamb in range(len(self.alpha)):
                        scores[ksubj, lamb, :] = np.diag(
                            np.corrcoef(yhat[..., lamb], yy, rowvar=False),
                            k=self.n_chans_,
                        )
            return scores

    def xfit(
        self,
        X,
        y,
        n_splits=5,
        lagged=False,
        drop=True,
        feat_names=(),
        plot=False,
        verbose=False,
    ):
        """Apply a cross-validation procedure to find the best regularisation parameters
        among the list of alphas given (ndim alpha must be == 1, and len(alphas)>1).
        If there are several subjects, will return a list of best alphas for each subjetc individually.
        User is expected to re-fit TRF fr each subject using their best individual alpha.

        For a single subject (y is 2-dimensional), the TRF stored is the one with best alpha.

        Notes
        -----
        The cross-validation procedure is a simple K-fold procedure with shuffling of samples.
        This is prone to some leakage since lags span several contiguous samples...
        """
        # Make sure we have several alphas
        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError(
                "Supply several alphas to TRF constructor to use this method."
            )

        self.fill_lags()

        y = np.asarray(y)
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(y) == 3 else y.nbytes
        estimated_mem_usage = (
            X.nbytes * (len(self.lags) if not lagged else 1) + y_memory
        )
        if estimated_mem_usage / 1024.0**3 > mem_check():
            raise MemoryError(
                "Not enough RAM available! (needed %.1fGB, but only %.1fGB available)"
                % (estimated_mem_usage / 1024.0**3, mem_check())
            )

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        n_samples_all = (
            y.shape[0] if y.ndim == 2 else y.shape[1]
        )  # this include non-valid samples for now

        if drop:
            self.valid_samples_ = np.logical_not(
                np.logical_or(
                    np.arange(n_samples_all) < abs(max(self.lags)),
                    np.arange(n_samples_all)[::-1] < abs(min(self.lags)),
                )
            )
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:, self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(
                X,
                lags=self.lags,
                mode="valid" if drop else "full",
                fill_value=np.nan if drop else 0.0,
                block_order=self.block_order,
            )

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Now cross-validation procedure:
        kf = KFold(n_splits=n_splits)
        if y.ndim == 2:  # single-subject
            scores = np.zeros((n_splits, 1, len(self.alpha), self.n_chans_))
            for kfold, (train, test) in enumerate(kf.split(X)):
                if verbose:
                    print("Training/Evaluating fold %d/%d" % (kfold + 1, n_splits))
                betas = _svd_regress(X[train, :], y[train, :], self.alpha)
                yhat = np.einsum("ij,jkl->ikl", X[test, :], betas)
                for lamb in range(len(self.alpha)):
                    scores[kfold, 0, lamb, :] = np.diag(
                        np.corrcoef(yhat[..., lamb], y[test, :], rowvar=False),
                        k=self.n_chans_,
                    )
        else:  # multi-subject
            scores = np.zeros((n_splits, y.shape[0], len(self.alpha), self.n_chans_))
            for kfold, (train, test) in enumerate(kf.split(X)):
                if verbose:
                    print("Training/Evaluating fold %d/%d" % (kfold + 1, n_splits))
                betas = _svd_regress(X[train, :], y[:, train, :], self.alpha)
                yhat = np.einsum("ij,jkl->ikl", X[test, :], betas)
                for ksubj, yy in enumerate(y[:, test, :]):
                    for lamb in range(len(self.alpha)):
                        scores[kfold, ksubj, lamb, :] = np.diag(
                            np.corrcoef(yhat[..., lamb], yy, rowvar=False),
                            k=self.n_chans_,
                        )

        # Best alpha
        peaks = scores.mean(0).mean(-1).argmax(1)

        # Plotting
        if plot:
            scores_toplot = scores.mean(0).mean(-1).T
            lines = plt.semilogx(self.alpha, scores_toplot)
            if y.ndim == 3:  # multi-subject (search best alpha PER subject)
                for k, p in enumerate(peaks):
                    plt.semilogx(
                        self.alpha[p],
                        scores_toplot[p, k],
                        "*",
                        ms=10,
                        color=lines[k].get_color(),
                    )
            else:
                plt.semilogx(
                    self.alpha[scores.mean(0).mean(-1).argmax()],
                    scores_toplot[scores.mean(0).mean(-1).argmax()],
                    "*k",
                    ms=10,
                )

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :, peaks[0]]
            betas = betas[1:, :, peaks[0]]

        self.coef_ = self._beta_to_coef(betas)
        self.fitted = True

        if y.ndim == 3:
            return scores, self.alpha[peaks]
        else:
            return scores, self.alpha[peaks[0]]

    def predict(self, X):
        """Compute output based on fitted coefficients and feature matrix X.

        Parameters
        ----------
        X : ndarray
            Matrix of features (can be already lagged or not).

        Returns
        -------
        ndarray
            Reconstruction of target with current beta estimates

        Notes
        -----
        If the matrix onky has features in its column (not yet lagged), the lagged version
        of the feature matrix will be created on the fly (this might take some time if the matrix
        is large).

        """
        assert self.fitted, "Fit model first!"
        betas = self._coef_to_beta(self.coef_)

        if self.fit_intercept:
            betas = np.r_[self.intercept_[:, None].T, betas]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            if self.verbose:
                LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(
                X,
                lags=self.lags,
                mode="full",
                fill_value=0.0,
                block_order=self.block_order,
            )
            # Adding intercept feature:
            if self.fit_intercept:
                X = np.hstack([np.ones((len(X), 1)), X])

        return X.dot(betas)

    def score(self, Xtest, ytrue, scoring="corr", reduce_multi=None):
        """Compute a score of the model given true target and estimated target from ``Xtest``.

        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used ("corr", "rmse", "mse")
        reduce_multi : None or callable or str
            The score by default return the score for each output (channel). However, sklearn pipelines
            for cross-validation might require a single number from the scorring function.
            This can be achieved by _reducing_ the scores either by taking the mean or the sum across channels
            (respectively with 'mean' or 'sum'). If a callable is used, its signature must be (1d-ndarray) -> float,
            similar to :func:`np.mean`.

        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == "corr":
            score = np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=self.n_chans_)
        elif scoring == "rmse":
            score = np.sqrt(np.mean((yhat - ytrue) ** 2, 0))
        elif scoring == "mse":
            score = np.mean((yhat - ytrue) ** 2, 0)
        else:
            raise NotImplementedError(
                "Only correlation score or (r)mse is valid for now..."
            )
        if reduce_multi is None:
            return score
        else:
            if isinstance(reduce_multi, str):
                return getattr(np, reduce_multi)(score)
            elif callable(reduce_multi):
                return reduce_multi(score)

    def apply_func(self, func):
        """
        Apply a function over all values in `coef_` and `intercept_`.

        Parameters
        ---------
        func : callable
            The funciton must take an array as an input and return an array of same size.

        Returns
        -------
        trf : TRFEstimator
            A new instance with transformed values.
        """
        trf = self.copy()
        trf.coef_ = func(self.coef_)
        assert trf.coef_.shape == self.coef_.shape, (
            f"{func} must apply to array and return an array of same shape."
        )
        trf.intercept_ = func(self.intercept_)
        return trf

    def plot(
        self,
        feat_id=None,
        ax=None,
        spatial_colors=False,
        info=None,
        picks=None,
        plot_kws={},
        **kwargs,
    ):
        """Plot the TRF of the feature requested as a *butterfly* plot.

        Parameters
        ----------
        feat_id : ``list`` | ``int``
            Index of the feature requested or list of features.
            Default is to use all features.
        ax : array of axes (flatten)
            list of subaxes
        plot_kws : ``**dict``
            Parameters to pass to :func:`plt.plot` (so that you can control the line style, color, etc...)
        **kwargs : ``**dict``
            Parameters to pass to :func:`plt.subplots`

        Returns
        -------
        fig : :class:`plt.Figure`
        """
        if isinstance(feat_id, int):
            feat_id = list(feat_id)  # cast into list to be able to use min, len, etc...
        if not feat_id:
            feat_id = range(self.n_feats_)

        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) < self.n_feats_]), (
            "Feat ids not in range"
        )

        if ax is None:
            if "figsize" not in kwargs:
                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=np.size(feat_id),
                    squeeze=False,
                    figsize=(
                        plt.rcParams["figure.figsize"][0] * np.size(feat_id),
                        plt.rcParams["figure.figsize"][1],
                    ),
                    **kwargs,
                )
                ax = ax.ravel()
            else:
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), **kwargs)
        else:
            if hasattr(ax, "__len__"):
                fig = ax[0].figure
            else:
                fig = ax.figure
                ax = [ax]

        if info is not None:
            info["sfreq"] = (
                self.srate
            )  # need that fix in case info is from some other processed data
            for k, feat in enumerate(feat_id):
                plot_interactive(
                    self.coef_[:, feat, :].T,
                    info=info,
                    ax=ax[k],
                    tmin=self.tmin,
                    picks=picks,
                )
                if self.feat_names_:
                    ax[k].set_title(f"{self.feat_names_[feat]:s}")
            return fig

        if spatial_colors:
            assert info is not None, (
                "To use spatial colouring, you must supply raw.info instance"
            )
            colors = get_spatial_colors(info)

        for k, feat in enumerate(feat_id):
            ax[k].plot(self.times, self.coef_[:, feat, :], **plot_kws)
            if self.feat_names_:
                ax[k].set_title(f"{self.feat_names_[feat]:s}")
            if spatial_colors:
                lines = ax[k].get_lines()
                for kc, l in enumerate(lines):
                    l.set_color(colors[kc])

        return fig

    def plot_topomap(self, time_lag, feat_id, info, ax=None, plot_kws={}, **kwargs):
        """Plot the topomap of the TRF at a given time-lag.

        Parameters
        ----------
        time_lag : ``float``
            Time-lag at which to plot the topomap.
        feat_id : ``int`` | ``str``
            Index of the feature requested. Can also be the name of the feature.
        info : :class:`mne.Info`
            Info instance from MNE.
        ax : :class:`plt.Axes`
            Axes to plot on.
        plot_kws : ``**dict``
            Parameters to pass to :func:`mne.viz.plot_topomap`
        **kwargs : ``**dict``
            Parameters to pass to :func:`mne.viz.plot_topomap`
        """
        assert self.fitted, "Fit the model first!"
        assert self.tmin <= time_lag <= self.tmax, "Time-lag not in range"
        if isinstance(feat_id, int):
            assert 0 <= feat_id < self.n_feats_, "Feat id not in range"
        if isinstance(feat_id, str):
            assert feat_id in self.feat_names_, (
                f"Features {feat_id} not in {self.feat_names_}"
            )
            feat_id = self.feat_names_.index(feat_id)
        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)
        else:
            fig = ax.figure
        from pyeeg.vizu import topomap

        topomap(
            self.coef_[np.argmin(np.abs(self.times - time_lag)), feat_id, :],
            info,
            ax=ax,
            **plot_kws,
        )
        return fig

    def _select_time_lag(self, indices):
        trf = self.copy()
        trf.coef_ = self.coef_[indices, :, :]
        trf.times = self.times[indices]
        trf.lags = self.lags[indices]
        return trf

    def _select_features(self, indices):
        selected_feature_alphas = (
            self.feature_alphas[indices] if self.feature_alphas is not None else None
        )
        trf = TRFEstimator(
            tmin=self.tmin,
            tmax=self.tmax,
            srate=self.srate,
            alpha=self.alpha,
            feature_alphas=selected_feature_alphas,
            block_order=self.block_order,
        )
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = self.feat_names_[indices] if self.feat_names_ else None
        trf.n_feats_ = len(indices)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_
        return trf

    def __getitem__(self, feats):
        """
        Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...).
        If a float, or an array of floats is supplied, will return a new TRF instance with only the corresponding time-lags.
        """
        # Argument check
        integer_indices = isinstance(feats, int) or (
            np.ndim(feats) > 0 and all([isinstance(f, int) for f in feats])
        )
        if isinstance(feats, (float, np.ndarray)):
            assert np.all([f >= self.tmin and f <= self.tmax for f in feats]), (
                "Time-lags not in range"
            )
            indices = np.argmin(np.abs(self.times - feats))
            return self._select_time_lag(indices)

        if self.feat_names_ is None or integer_indices:
            if np.ndim(feats) > 0:
                assert isinstance(feats[0], int), (
                    "Type not understood, feat_names are ot defined, can only index with int"
                )
                indices = feats

            else:
                assert isinstance(feats, int), (
                    "Type not understood, feat_names are ot defined, can only index with int"
                )
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]), (
                    "an element in argument %s in not present in %s"
                    % (feats, self.feat_names_)
                )
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s" % (
                    feats,
                    self.feat_names_,
                )
                indices = [self.feat_names_.index(feats)]
                feats = [feats]
        return self._select_features(indices)

    def __repr__(self):
        tmin = self.tmin if self.tmin is not None else float("nan")
        tmax = self.tmax if self.tmax is not None else float("nan")
        if self.fitted:
            obj = """TRFEstimator(
                alpha=%s,
                fit_intercept=%s,
                srate=%d,
                tmin=%.2f
                tmax=%.2f,
                n_feats=%d,
                n_chans=%d,
                n_lags=%d,
                features : %s
            )
            """ % (
                self.alpha,
                self.fit_intercept,
                self.srate,
                tmin,
                tmax,
                self.n_feats_,
                self.n_chans_,
                len(self.lags),
                str(self.feat_names_),
            )
            return obj
        else:
            obj = """TRFEstimator(
                alpha=%s,
                fit_intercept=%s,
                srate=%d,
                tmin=%.2f
                tmax=%.2f,
                
                Not fitted yet.
            )
            """ % (
                self.alpha,
                self.fit_intercept,
                self.srate,
                tmin,
                tmax,
            )
            return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (
            other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ == self.n_chans_
        ), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(
            tmin=self.tmin,
            tmax=self.tmax,
            srate=self.srate,
            alpha=self.alpha,
            block_order=self.block_order,
        )
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf

    def __truediv__(self, scalar):
        "Make available the '/' operator. Will simply divide coefficients by scalar (useful for averaging)."
        assert isinstance(scalar, (int, float)), "Can only divide by scalar"
        assert scalar != 0, "Cannot divide by zero"
        trf = self.copy()
        trf.coef_ = trf.coef_ / scalar
        trf.intercept_ = trf.intercept_ / scalar
        return trf

    def copy(self):
        trf = TRFEstimator(
            tmin=self.tmin,
            tmax=self.tmax,
            srate=self.srate,
            alpha=self.alpha,
            feature_alphas=self.feature_alphas,
            block_order=self.block_order,
        )
        trf.coef_ = self.coef_
        trf.intercept_ = self.intercept_
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        return trf

    def save(self, filename):
        """
        Save the current trf object to file.
        Format used is Numpy's binary npz.

        Parameters
        ----------
        filename : str
            Full path name.

        Returns
        -------
        None.

        Raise
        -----
        AssertionError: if trf is empty (not fitted).
        """
        assert self.fitted, "Fit TRF before saving it."
        trf = {
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "feat_names_": self.feat_names_,
            "srate": self.srate,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "times": self.times,
            "alpha": self.alpha,
            "feature_alphas": self.feature_alphas,
        }
        np.savez(filename, **trf)

    def load(filename):
        """
        Load and return a TRF instance from numpy archive file (created with trf.save)

        Parameters
        ----------
        filename : str
            Full path name.

        Returns
        -------
        TRFEstimator instance
        """
        npzdata = np.load(filename, allow_pickle=True)
        feature_alphas = None
        if "feature_alphas" in npzdata:
            stored_feature_alphas = npzdata["feature_alphas"]
            if (
                stored_feature_alphas.ndim != 0
                or stored_feature_alphas.item() is not None
            ):
                feature_alphas = stored_feature_alphas
        trf = TRFEstimator(
            tmin=npzdata["tmin"],
            tmax=npzdata["tmax"],
            srate=npzdata["srate"],
            alpha=npzdata["alpha"],
            feature_alphas=feature_alphas,
            block_order="lags",
        )
        trf.fill_lags()
        trf.intercept_ = npzdata["intercept_"]
        trf.feat_names_ = npzdata["feat_names_"]
        trf.coef_ = npzdata["coef_"]
        trf.n_chans_ = trf.coef_.shape[-1]
        trf.n_feats_ = trf.coef_.shape[1]
        trf.fitted = True
        return trf
