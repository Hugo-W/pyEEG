# -*- coding: utf-8 -*-
"""
Utilities
---------

Desctription
~~~~~~~~~~~~

This module contains some utilities functions for data manipulation in general,
signal processing or/and text processing.
For instance function to create a matrix of lagged-time series for input to our
forward and backward models.

"""

#### Libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

def lag_matrix(data, lag_samples=(-1, 0, 1), filling=np.nan, drop_missing=False):
    """Helper function to create a matrix of lagged time series.

    The lag can be arbitrarily spaced. Check other functions to create series of lags
    whether they are contiguous or sparsely spanning a time window :func:`lag_span` and
    :func:`lag_sparse`.

    Parameters
    ----------
    data : ndarray (nsamples x nfeats)
        Multivariate data
    lag_samples : list
        Shift in _samples_ to be applied to data. Negative shifts are lagged in the past,
        positive shits in the future, and a shift of 0 represents the data array as it is
        in the input `data`.
    filling : float
        What value to use to fill entries which are not defined (Default: NaN).
    drop_missing : bool
        Whether to drop rows where filling occured.

    Returns
    -------
    lagged : ndarray (nsamples_new x nfeats*len(lag_samples))
        Matrix of lagged time series.

    Example
    -------
    >>> data = np.asarray([[1,2,3,4,5,6],[7,8,9,10,11,12]]).T
    >>> out = lag_matrix(data, (0,1))
    >>> out
    array([[ 1.,  7.,  2.,  8.],
            [ 2.,  8.,  3.,  9.],
            [ 3.,  9.,  4., 10.],
            [ 4., 10.,  5., 11.],
            [ 5., 11.,  6., 12.],
            [ 6., 12., nan, nan]])

    """
    dframe = pd.DataFrame(data)

    cols = []
    for lag in lag_samples:
        cols.append(dframe.shift(-lag))

    dframe = pd.concat(cols, axis=1)
    dframe.fillna(filling, inplace=True)
    if drop_missing:
        dframe.dropna(inplace=True)

    return dframe.get_values()

def lag_span(tmin, tmax, srate=125):
    """Create an array of lags spanning the time window [tmin, tmax].

    Parameters
    ----------
    tmin : float
        In seconds
    tmax : float
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    sample_min, sample_max = int(np.ceil(tmin * srate)), int(np.ceil(tmax * srate))
    return np.arange(sample_min, sample_max)

def lag_sparse(times, srate=125):
    """Create an array of lags for the requested time point in `times`.

    Parameters
    ----------
    times : list
        List of time point in seconds
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    return np.asarray([int(np.ceil(t * srate)) for t in times])
