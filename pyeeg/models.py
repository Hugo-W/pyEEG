# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,wrong-import-position
"""
In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)
    - CCA (in cca.py)
    - ...

TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.

"""
import logging
import numpy as np
from scipy import stats
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse, mem_check
from .mcca import mCCA

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

def _svd_regress(x, y, alpha=0.):
    """Linear regression using svd.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats)
    y : ndarray (nsamples, nchans) or list of such
        If a list of such arrays is given, each element of the
        list is treated as an individual subject

    Returns
    -------
    betas : ndarray (nfeats, nchans)
        Coefficients

    Raises
    ------
    ValueError
        If alpha < 0 (coefficient of L2 - regularization)
    NotImplementedError
        If len(alpha) > 1 (later will do several fits)

    Notes
    -----
    A warning is shown in the case where nfeats > nsamples, if so the user
    should rather use partial regression.
    """
    try:
        assert np.size(alpha) == 1, "Alpha cannot be an array (in the future yes)"
    except AssertionError:
        raise NotImplementedError
    try:
        assert alpha >= 0, "Alpha must be positive"
    except AssertionError:
        raise ValueError

    [U, s, V] = np.linalg.svd(x, full_matrices=False)
    if np.ndim(y) == 3:
        Uty = np.zeros((U.shape[1], y.shape[2]))
        for Y in y:
            Uty += U.T @ Y
        Uty /= len(y)
    else:
        Uty = U.T @ y
    Vsreg = V.T @ np.diag(s/(s**2 + alpha))
    betas = Vsreg @ Uty
    return betas

class TRFEstimator(BaseEstimator):
    """Temporal Response Function (TRF) Estimator Class.

    This class allows to estimate TRF from a set of feature signals and an EEG dataset in the same fashion
    than ReceptiveFieldEstimator does in MNE.
    However, an arbitrary set of lags can be given. Namely, it can be used in two ways:

    - calling with `tmin` and tmax` arguments will compute lags spanning from `tmin` to `tmax`
    - with the `times` argument, one can request an arbitrary set of time lags at which to compute
    the coefficients of the TRF

    Attributes
    ----------
    lags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    use_reularisation : bool
        Whether ot not the Ridge regularisation is used to compute the TRF
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    fitted : bool
        True once the TRF has been fitted on EEG data
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features

    Notes
    -----
        - Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data (i.e. after
        the method :meth:`TRFEstimator.fit` has been called).
        - Can fit on a list of multiple dataset, where we have a list of target Y and
        a single stimulus matrix of features X, then the computation is made such that
        the coefficients computed are similar to those obtained by concatenating all matrices

    Examples
    --------
    >>> trf = TRFEstimator(tmin=-0.5, tmax-1.2, srate=125)
    >>> x = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000, 2)
    >>> trf.fit(x, y, lagged=False)
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1.,
                 alpha=0., fit_intercept=True):

        if tmin and tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(tmin, tmax, srate=srate)[::-1] #pylint: disable=invalid-unary-operand-type
            #self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / srate
        else:
            self.times = np.asarray(times)
            self.lags = lag_sparse(self.times, srate)[::-1]

        self.srate = srate
        self.alpha = alpha
        self.use_regularisation = alpha > 0.
        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.rotations_ = None # matrices to be used to rotate coefficients into a 'better conditonned subspace'
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None

    def fit(self, X, y, lagged=False, drop=True, feat_names=(), rotations=()):
        """Fit the TRF model.

        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
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

        Returns
        -------
        coef_ : ndarray (nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """
        y = np.asarray(y)
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)"%(estimated_mem_usage/1024.**3, mem_check()))

        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags)
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names
        
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1] # this include non-valid samples for now

        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)

        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:, self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=drop, filling=np.nan if drop else 0.)
        #else: # simply do the dropping assuming it hasn't been done when default values are supplied
        #    X = X[self.valid_samples_, :]
        '''
        if not lagged:
            if drop:
                X = lag_matrix(X, lag_samples=self.lags, drop_missing=True)

                # Droping rows of NaN values in y
                if any(np.asarray(self.lags) < 0):
                    drop_top = abs(min(self.lags))
                    y = y[drop_top:, :] if y.ndim == 2 else y[:, drop_top:, :]
                if any(np.asarray(self.lags) > 0):
                    drop_bottom = abs(max(self.lags))
                    y = y[:-drop_bottom, :] if y.ndim == 2 else y[:, :-drop_bottom, :]
            else:
                X = lag_matrix(X, lag_samples=self.lags, filling=0.)
        '''
        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Solving with svd or least square:
        if self.use_regularisation or np.ndim(y) == 3:
            # svd method:
            betas = _svd_regress(X, y, self.alpha)
        else:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Get t-statistic and p-vals if regularization is ommited
        if not self.use_regularisation:
            cov_betas = X.T @ X
            H = X @ np.inv(cov_betas) @ X.T
            if np.ndim(y) == 3:
                dof = sum(list(map(len, y))) - (len(betas)+1)
                sigma = 0.
                for yy in y:
                    sigma += yy.T @ (np.eye(self.n_chans_) - H) @ yy
                sigma /= dof
            else:
                dof = len(y)-(len(betas)+1)
                sigma = y.T @ (np.eye(y.shape[1]) - H) @ y / dof
            C = sigma * np.inv(cov_betas)
            self.tvals_ = betas / np.sqrt(np.diag(C))
            self.pvals_ = 2 * (1-stats.t.cdf(self.tvals_, df=dof))

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        if rotations:
            newbetas = np.zeros((len(self.lags) * self.n_feats_, self.n_chans_))
            for k, rot in zip(range(self.n_feats_), rotations):
                if not rot: rot = np.eye(self.lags)
                newbetas[::self.n_feats_, :] = rot @ betas[...]
            betas = newbetas

        self.coef_ = np.reshape(betas, (len(self.lags), self.n_feats_, self.n_chans_))
        self.coef_ = self.coef_[::-1, :, :] # need to flip the first axis of array to get correct lag order
        self.fitted = True

        return self

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
        betas = np.reshape(self.coef_[::-1, :, :], (len(self.lags) * self.n_feats_, self.n_chans_))

        if self.fit_intercept:
            betas = np.r_[self.intercept_[:, None].T, betas]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)

        return X.dot(betas)

    def score(self, Xtest, ytrue, scoring="corr"):
        """Compute a score of the model given true target and estimated target from Xtest.

        Parameters
        ----------
        Xtest : ndarray
            Array used to get "yhat" estimate from model
        ytrue : ndarray
            True target
        scoring : str (or func in future?)
            Scoring function to be used

        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=self.n_chans_)
        else:
            raise NotImplementedError("Only correlation score is valid for now...")
        

    def plot(self, feat_id=None, **kwargs):
        """Plot the TRF of the feature requested as a *butterfly* plot.

        Parameters
        ----------
        feat_id : list or int
            Index of the feature requested or list of features.
            Default is to use all features.
        **kwargs : **dict
            Parameters to pass to :func:`plt.subplots`
        """
        if isinstance(feat_id, int):
            feat_id = list(feat_id) # cast into list to be able to use min, len, etc...
        if not feat_id:
            feat_id = range(self.n_feats_)
        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) < self.n_feats_]), "Feat ids not in range"

        _, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), squeeze=False, **kwargs)

        for k, feat in enumerate(feat_id):
            ax[0, k].plot(self.times, self.coef_[:, feat, :])
            if self.feat_names_:
                ax[0, k].set_title('TRF for {:s}'.format(self.feat_names_[feat]))
