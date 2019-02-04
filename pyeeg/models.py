# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Desctription
~~~~~~~~~~~~

In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)
    - CCA
    - ?

TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.

"""

import logging
import numpy as np
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

class TRFEstimator(BaseEstimator):
    """TODO: fill in docstring
    """

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., 
                 alpha=0., fit_intercept=True):

        if tmin and tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(tmin, tmax, srate=srate)
            self.times = self.lags / srate
        else:
            self.lags = lag_sparse(times, srate)
            self.times = np.asarray(times)
        
        self.srate = srate
        self.alpha = alpha
        self.use_regularisation = alpha > 0.
        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None

    def fit(self, X, y, drop=True):
        """Fit the TRF model.

        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
        y : ndarray (nsamples x nchans)
            EEG data

        Returns
        -------
        coef_ : ndarray (nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """
        self.n_feats_ = X.shape[1]
        self.n_chans_ = y.shape[1]
        if drop:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=True)
            # Droping rows of NaN values
            if any(np.asarray(self.lags) < 0):
                drop_top = abs(min(self.lags))
                y = y[drop_top:, :]
            if any(np.asarray(self.lags) > 0):
                drop_bottom = abs(max(self.lags))
                y = y[:-drop_bottom, :]
        else:
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)

        if self.use_regularisation:
            # svd method:
            [U, s, V] = np.linalg.svd(X, full_matrices=False)
            Uty = U.T @ y
            Vsreg = V @ np.diag(s/(s**2 + self.alpha))
            betas = Vsreg @ Uty
        else:
            betas = np.linalg.lstsq(X, y)

        if self.fit_intercept:
            self.intercept_ = betas[0, :]
            betas = betas[1:, :]

        self.coef_ = np.reshape(betas, len(self.lags), self.n_feats_)
        self.fitted = True
