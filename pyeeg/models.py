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
from sklearn.model_selection import KFold
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse, mem_check
from .vizu import get_spatial_colors

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__.split('.')[0])

def _svd_regress(x, y, alpha=0.):
    """Linear regression using svd.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats)
    y : ndarray (nsamples, nchans) or list of such
        If a list of such arrays is given, each element of the
        list is treated as an individual subject
    alpha : float or array-like
        If array, will compute betas for every regularisation parameters at once

    Returns
    -------
    betas : ndarray (nfeats, nchans, len(alpha))
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
    # cast alpha in ndarray
    if isinstance(alpha, float):
        alpha = np.asarray([alpha])
    else:
        alpha = np.asarray(alpha)

    try:
        assert np.all(alpha >= 0), "Alpha must be positive"
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
    # Cast all alpha (regularization param) in a 3D matrix,
    # each slice being a diagonal matrix of s/(s**2+lambda)
    eigenvals_scaled = np.zeros((*V.shape, np.size(alpha)))
    eigenvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(s[:, None], np.size(alpha), axis=1) / \
        (np.repeat(s[:, None]**2, np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
    # A dot product instead of matmul allows to repeat multiplication alike across third dimension (alphas)
    Vsreg = np.dot(V.T, eigenvals_scaled) #np.diag(s/(s**2 + alpha))
    # Using einsum to control which access get multiplied, again leaving alpha's dimension "untouched"
    betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty) #Vsreg @ Uty
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
        self.alpha = alpha
        if np.ndim(alpha) == 0:
            self.use_regularisation = alpha > 0.
        else:
            self.use_regularisation = np.any(np.asarray(alpha) > 0.)
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.rotations_ = None # matrices to be used to rotate coefficients into a 'better conditonned subspace'
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        # The two following are only defined if simple least-square (no reg.) is used
        self.tvals_ = None
        self.pvals_ = None

    def fill_lags(self):
        """Fill the lags attributes.

        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.
        
        """
        if self.tmin and self.tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[::-1] #pylint: disable=invalid-unary-operand-type
            #self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]
        

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
        self.fill_lags()

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
            betas = _svd_regress(X, y, self.alpha)[..., 0]
        else:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        if rotations:
            newbetas = np.zeros((len(self.lags) * self.n_feats_, self.n_chans_))
            for _, rot in zip(range(self.n_feats_), rotations):
                if not rot:
                    rot = np.eye(self.lags)
                newbetas[::self.n_feats_, :] = rot @ betas[...]
            betas = newbetas

        self.coef_ = np.reshape(betas, (len(self.lags), self.n_feats_, self.n_chans_))
        self.coef_ = self.coef_[::-1, :, :] # need to flip the first axis of array to get correct lag order
        self.fitted = True

        # Get t-statistic and p-vals if regularization is ommited
        if not self.use_regularisation:
            cov_betas = X.T @ X
            # Compute variance sigma (MSE)
            if np.ndim(y) == 3:
                dof = sum(list(map(len, y))) - (len(betas)+1+1) # 1 for mean, 1 for intercept (betas does not contains it)
                sigma = 0.
                for yy in y:
                    sigma += np.sum((yy - self.predict(X))**2, axis=0)
                sigma /= dof
            else:
                dof = len(y)-(len(betas)+1+1)
                sigma = np.sum((y - self.predict(X))**2, axis=0) / dof
            # Covariance matrix on betas
            #C = sigma * np.linalg.inv(cov_betas)
            C = np.einsum('ij,k', np.linalg.inv(cov_betas), sigma)
            # Actual stats
            self.tvals_ = betas / np.sqrt(C.diagonal(axis1=0, axis2=1).swapaxes(0, 1)[1:, :])
            self.pvals_ = 2 * (1-stats.t.cdf(abs(self.tvals_), df=dof))

        return self

    def xfit(self, X, y, n_splits=5, lagged=False, drop=True, feat_names=(), plot=False, verbose=False):
        """Apply a cross-validation procedure to find the best regularisation parameters
        among the list of alphas given (ndim alpha must be == 1, and len(alphas)>1).
        If there are several subjects, will return a list of best alphas for each subjetc individually.
        User is expected to re-fit TRF fr each subject using their best individual alpha.

        For a singly subject (y 2-dimensional), the TRF stored is the one with best alpha.

        Notes
        -----
        The cross-validation procedure is a simple K-fold procedure with shuffling of samples.
        This is prone to some leakage since lags span several contiguous samples...
        """
        # Make sure we have several alphas
        if np.ndim(self.alpha) < 1 or len(self.alpha) <= 1:
            raise ValueError("Supply several alphas to TRF constructor to use this method.")

        self.fill_lags()

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
        
        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])        

        # Now cross-validation procedure:
        kf = KFold(n_splits=n_splits)
        if y.ndim == 2: # single-subject
            scores = np.zeros((n_splits, 1, len(self.alpha), self.n_chans_))
            for kfold, (train, test) in enumerate(kf.split(X)):
                if verbose: print("Training/Evaluating fold %d/%d"%(kfold+1, n_splits))
                betas = _svd_regress(X[train, :], y[train, :], self.alpha)
                yhat = np.einsum('ij,jkl->ikl', X[test, :], betas)
                for lamb in range(len(self.alpha)):
                    scores[kfold, 0, lamb, :] = np.diag(np.corrcoef(yhat[..., lamb], y[test, :], rowvar=False), k=self.n_chans_)
        else: # multi-subject
            scores = np.zeros((n_splits, y.shape[0], len(self.alpha), self.n_chans_))
            for kfold, (train, test) in enumerate(kf.split(X)):
                if verbose: print("Training/Evaluating fold %d/%d"%(kfold+1, n_splits))
                betas = _svd_regress(X[train, :], y[:, train, :], self.alpha)
                yhat = np.einsum('ij,jkl->ikl', X[test, :], betas)
                for ksubj, yy in enumerate(y[:, test, :]):
                    for lamb in range(len(self.alpha)):
                        scores[kfold, ksubj, lamb, :] = np.diag(np.corrcoef(yhat[..., lamb], yy, rowvar=False), k=self.n_chans_)

        # Best alpha
        peaks = scores.mean(0).mean(-1).argmax(1)

        # Plotting
        if plot:
            scores_toplot = scores.mean(0).mean(-1).T
            lines = plt.semilogx(self.alpha, scores_toplot)
            if y.ndim == 3: # multi-subject (search best alpha PER subject)
                for k, p in enumerate(peaks):
                    plt.semilogx(self.alpha[p], scores_toplot[p, k], '*', ms=10, color=lines[k].get_color())
            else:
                plt.semilogx(self.alpha[scores.mean(0).mean(-1).argmax()],
                            scores_toplot[scores.mean(0).mean(-1).argmax()], '*k', ms=10,)
            
        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = betas[0, :, peaks[0]]
            betas = betas[1:, :, peaks[0]]

        self.coef_ = np.reshape(betas, (len(self.lags), self.n_feats_, self.n_chans_))
        self.coef_ = self.coef_[::-1, :, :] # need to flip the first axis of array to get correct lag order
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
        betas = np.reshape(self.coef_[::-1, :, :], (len(self.lags) * self.n_feats_, self.n_chans_))

        if self.fit_intercept:
            betas = np.r_[self.intercept_[:, None].T, betas]

        # Check if input has been lagged already, if not, do it:
        if X.shape[1] != int(self.fit_intercept) + len(self.lags) * self.n_feats_:
            LOGGER.info("Creating lagged feature matrix...")
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
            # Adding intercept feature:
            if self.fit_intercept:
                X = np.hstack([np.ones((len(X), 1)), X])

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
            Scoring function to be used ("corr", "rmse")

        Returns
        -------
        float
            Score value computed on whole segment.
        """
        yhat = self.predict(Xtest)
        if scoring == 'corr':
            return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=self.n_chans_)
        elif scoring == 'rmse':
            return np.sqrt(np.mean((yhat-ytrue)**2, 0))
        else:
            raise NotImplementedError("Only correlation score is valid for now...")


    def plot(self, feat_id=None, ax=None, spatial_colors=False, info=None, **kwargs):
        """Plot the TRF of the feature requested as a *butterfly* plot.

        Parameters
        ----------
        feat_id : list or int
            Index of the feature requested or list of features.
            Default is to use all features.
        ax : array of axes (flatten)
            list of subaxes
        **kwargs : **dict
            Parameters to pass to :func:`plt.subplots`

        Returns
        -------
        fig : figure
        """
        if isinstance(feat_id, int):
            feat_id = list(feat_id) # cast into list to be able to use min, len, etc...
            if ax is not None:
                fig = ax.figure
        if not feat_id:
            feat_id = range(self.n_feats_)
        if len(feat_id) > 1:
            if ax is not None:
                fig = ax[0].figure
        assert self.fitted, "Fit the model first!"
        assert all([min(feat_id) >= 0, max(feat_id) < self.n_feats_]), "Feat ids not in range"

        if ax is None:
            if 'figsize' not in kwargs.keys():
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), figsize=(plt.rcParams['figure.figsize'][0] * np.size(feat_id), plt.rcParams['figure.figsize'][1]), **kwargs)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=np.size(feat_id), **kwargs)

        if spatial_colors:
            assert info is not None, "To use spatial colouring, you must supply raw.info instance"
            colors = get_spatial_colors(info)
            
        for k, feat in enumerate(feat_id):
            if len(feat_id) == 1:
                ax.plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax.set_title('TRF for {:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax.get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
            else:
                ax[k].plot(self.times, self.coef_[:, feat, :])
                if self.feat_names_:
                    ax[k].set_title('{:s}'.format(self.feat_names_[feat]))
                if spatial_colors:
                    lines = ax[k].get_lines()
                    for kc, l in enumerate(lines):
                        l.set_color(colors[kc])
                        
        return fig

    def __getitem__(self, feats):
        "Extract a sub-part of TRF instance as a new TRF instance (useful for plotting only some features...)"
        # Argument check
        if self.feat_names_ is None:
            if np.ndim(feats) > 0:
                assert isinstance(feats[0], int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = feats

            else:
                assert isinstance(feats, int), "Type not understood, feat_names are ot defined, can only index with int"
                indices = [feats]
                feats = [feats]
        else:
            if np.ndim(feats) > 0:
                assert all([f in self.feat_names_ for f in feats]), "an element in argument %s in not present in %s"%(feats, self.feat_names_)
                indices = [self.feat_names_.index(f) for f in feats]
            else:
                assert feats in self.feat_names_, "argument %s not present in %s"%(feats, self.feat_names_)
                indices = [self.feat_names_.index(feats)]
                feats = [feats]

        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax, srate=self.srate, alpha=self.alpha)
        trf.coef_ = self.coef_[:, indices]
        trf.feat_names_ = feats
        trf.n_feats_ = len(feats)
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags
        trf.intercept_ = self.intercept_

        return trf

    def __repr__(self):
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
        """%(self.alpha, self.fit_intercept, self.srate, self.tmin, self.tmax,
             self.n_feats_, self.n_chans_, len(self.lags), str(self.feat_names_))
        return obj

    def __add__(self, other_trf):
        "Make available the '+' operator. Will simply add coefficients. Be mindful of dividing by the number of elements later if you want the true mean."
        assert (other_trf.n_feats_ == self.n_feats_ and other_trf.n_chans_ == self.n_chans_), "Both TRF objects must have the same number of features and channels"
        trf = TRFEstimator(tmin=self.tmin, tmax=self.tmax, srate=self.srate, alpha=self.alpha)
        trf.coef_ = np.sum([self.coef_, other_trf.coef_], 0)
        trf.intercept_ = np.sum([self.intercept_, other_trf.intercept_], 0)
        trf.feat_names_ = self.feat_names_
        trf.n_feats_ = self.n_feats_
        trf.n_chans_ = self.n_chans_
        trf.fitted = True
        trf.times = self.times
        trf.lags = self.lags

        return trf
