# -*- coding: utf-8 -*-
"""
Module containing some preprocessing helper and function to be applied to EEG...

"""

#### Libraries
# Standard library
import numpy as np
from scipy import signal
from scipy.linalg import eigh as geigh
from joblib import Parallel, delayed
#import matplotlib.pyplot as plt
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance

# My libraries

# Mapping different estimator on the sklearn toolbox
def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X)
    return C

def _cov(X):
    "Wrapper for numpy cov estimator"
    return np.cov(X, rowvar=False)

def _corr(X):
    "Wrapper for numpy cov estimator"
    return np.corrcoef(X, rowvar=False)

def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': _cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': _corr
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est

def covariances(X, estimator='cov'):
    """Estimation of covariance matrices."""
    est = _check_est(estimator)
    assert X.ndim == 3, "Data must be 3d (trials, samples, channels)"
    Ntrials, Nsamples, Nchans = X.shape
    covmats = np.zeros((Ntrials, Nchans, Nchans))
    for i in range(Ntrials):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats

def covariance(X, estimator='cov'):
    """Estimation of one covariance matrix on whole dataset"""
    est = _check_est(estimator)
    if X.ndim == 3:
        nchans = X.shape[2]
        X = X.reshape((-1, nchans))
    return est(X)


def covariances_extended(X, P, estimator='cov'):
    """Special form covariance matrix where data are appended with another set."""
    est = _check_est(estimator)
    Ntrials, Nsamples, Nchans = X.shape
    Nsamples, Np = P.shape
    covmats = np.zeros((Ntrials, Nchans + Np, Nchans + Np))
    for i in range(Ntrials):
        covmats[i, :, :] = est(np.concatenate((P, X[i, :, :]), axis=0))
    return covmats

def create_filterbank(freqs, srate, filtertype=signal.cheby2, **kwargs):
    """Creates a filter bank, by default of chebychev type 2 filters.
    Parameters of filter are to be defined as name value pair arguments.
    Frequency bands are defined with boundaries instead of center frequencies.
    """
    normalized_freqs = np.asarray(freqs)/(srate/2.) # create normalized frequencies specifications
    return [filtertype(**kwargs, Wn=ff) for ff in normalized_freqs]

def apply_filterbank(data, fbank, filt_func=signal.lfilter, n_jobs=-1, axis=-1):
    """Applies a filterbank to a given multi-channel signal.

    Parameters
    ----------
    data : ndarray (samples, nchannels)
    fb : list
        list of (b,a) tuples, where b and a specify a digital filter

    Returns
    -------
    y : ndarray (nfilters, samples, nchannels)
    """
    return np.asarray(Parallel(n_jobs=n_jobs)(delayed(filt_func)(b, a, data, axis=axis) for b, a in fbank))

def get_power(signals, decibels=False, win=125, axis=-1, n_jobs=-1):
    """
    Compute the (log) power modulation of a signal by taking the smooth moving average of its square values.

    Parameters
    ----------
    signals : ndarray (nsamples, nchans)
        Input signals
    decibels : bool
        If True, will take the log power (default False).
    win : int
        Length of smoothing window for moving average (default 125) in samples.
    axis : int
        Axis on which to apply the transform
    n_jobs : int
        Number of cores to be used (Parrallel job).

    Returns
    -------
    out : ndarray (nsamples, nchans)
    """
    if axis != -1:
        signals = np.moveaxis(signals, axis, -1)
    nfreqs, nchans, _ = signals.shape
    out = np.zeros_like(signals)
    for k in range(nfreqs):
        feat = np.array(Parallel(n_jobs=n_jobs)(delayed(signal.convolve)(signals[k, i, :]**2, signal.windows.boxcar(win), 'same') for i in range(nchans)))
        if decibels:
            feat = np.log(feat + 1e-16)
        out[k, :, :] = feat

    return out

class MultichanWienerFilter():
    '''
    This class implements a multichannel Wiener Filter for artifact removal.
    The method is detailed in the reference paper *A generic EEG artifact removal algorithm based on the multi-channel
    Wiener filter* from Ben Somers et. al.

    To correctly train the model, one must supply portions of contaminated data and clean data. This can be selected visually
    using the annotation tool from MNE for instance, or automatically by detecting above threshold values and considering this as
    bad portions. It is ok to have large windows around bad data segments, however the clean segments must be artifact free.

    The model expects zero-mean data for both noisy and clean segments.

    Attributes
    ----------
        lags : list
            Lags used for general model (NOT IMPLEMENTED YET)
        low_rank : bool
            Whether to use low-rank approximation of covariance matrix for the artifactual data
        thresh : int or float
            If int, this will correspond to the rank prior
            If float, it will be considered as the percent of variance to be kept
        W_ : ndarray
            Once fitted, contains the filter coefficients

    Example
    -------
    TODO: Add code example
    Here is an example output:
    .. image:: img/MWF_EOG_cleaning_example.png
        :width: 400
    '''
    def __init__(self, lags=(0,), low_rank=False, thresh=None):
        self.lags = lags
        self.low_rank = low_rank
        self.thresh = thresh
    
    def fit(self, y_clean, y_artifact, cov_data=False):
        '''
        Fit model to data.

        Parameters
        ----------
        y_clean : ndarray
            Clean segments
        y_artifact : ndarray
            Artifact-contaminated segments
        cov_data : bool
            Whether the input data are already covariance matrices estimate for each class
        '''
        if cov_data:
            Sc = y_clean
            Sy = y_artifact
        else:
            Sc = covariance(y_clean) # neural covariance estimate
            Sy = covariance(y_artifact) # mixture covariance estimate
        if self.low_rank:
            print("Using low rank approximation")
            w, v = geigh(Sy, Sc)
            w, v = w[::-1], v[:, ::-1]
            sigma_y = np.diag(v.T @ Sy @ v)
            sigma_c = np.diag(v.T @ Sc @ v)
            sigma_d = sigma_y - sigma_c
            sigma_d[sigma_d <= 0] = 0.
            if self.thresh:
                if self.thresh >= 1:
                    print("Prior on rank of artifacts: %d"%self.thresh)
                    sigma_d[self.thresh:] = 0.
                else:
                    print("Keeping %.1f per cent of eigenvalues"%(self.thresh * 100))
                    #sigma_d[np.cumsum(sigma_d / sum(sigma_d)) > self.thresh] = 0.
                    sigma_d[int(self.thresh * len(sigma_d)):] = 0.
            Sd = np.linalg.inv(v.T) @ np.diag(sigma_d) @ np.linalg.inv(v)
        else:
            Sd = Sy - Sc # noise/artifact covariance estimate
        self.W_ = np.linalg.inv(Sy) @ Sd
        return self
    
    def transorm(self, x):
        '''
        Filter the data to remove artifact learned by the model.

        Parameters
        ----------
        x : data
            EEG data
        
        Returns
        -------
        Filtered EEG data
        '''
        return x - x @ self.W_
        
    def fit_transform(self, y_clean, y_artifact, x, cov_data=False):
        '''
        Train the model on input and transform directly the data in `x`.
        '''
        self.fit(y_clean, y_artifact, cov_data=cov_data)
        return self.transorm(x)
