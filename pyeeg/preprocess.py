# -*- coding: utf-8 -*-
"""
Module containing some preprocessing helper and function to be applied to EEG...

"""

#### Libraries
# Standard library
import numpy as np
from scipy import signal
from joblib import Parallel, delayed
#import matplotlib.pyplot as plt
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance

# My libraries

# Mapping different estimator on the sklearn toolbox
def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T)
    return C


def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': np.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': np.corrcoef
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
    """Estimation of covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    covmats = np.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i, :, :] = est(X[i, :, :])
    return covmats


def covariances_extended(X, P, estimator='cov'):
    """Special form covariance matrix where data are appended with another set."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = numpy.zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
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
