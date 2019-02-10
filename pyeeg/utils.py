# -*- coding: utf-8 -*-
"""
This module contains some utilities functions for data manipulation in general,
signal processing or/and text processing.
For instance function to create a matrix of lagged-time series for input to our
forward and backward models.

Also, utilities functions to apply _rolling_ methods (rolling average, variance, or so on)
wiht the NumPy **side trick** see `the Python Cookbook recipee`_ for more details.

.. _the Python Cookbook recipee: https://ipython-books.github.io/47-implementing-an-efficient-rolling-average-algorithm-with-stride-tricks/
"""

#### Libraries
import psutil
import numpy as np
from numpy.lib.stride_tricks import as_strided
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
        #cols.append(dframe.shift(-lag))
        cols.append(dframe.shift(lag))

    dframe = pd.concat(cols, axis=1)
    dframe.fillna(filling, inplace=True)
    if drop_missing:
        dframe.dropna(inplace=True)

    return dframe.get_values()
    #return dframe.loc[:, ::-1].get_values()

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
    #sample_min, sample_max = int(np.ceil(-tmax * srate)), int(np.ceil(-tmin * srate))
    #return np.arange(sample_max, sample_min, -1)

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

def _is_1d(arr):
    "Short utility function to check if an array is vector-like"
    return np.product(arr.shape) == max(arr.shape)

def is_pos_def(A):
    """Check if matrix is positive definite
    
    Ref: https://stackoverflow.com/a/44287862/5303618
    """
    if np.array_equal(A, A.conj().T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def shift_array(arr, win=2, overlap=0, padding=False, axis=0):
    """Returns segments of an array (overlapping moving windows)
    using the `as_strided` function from NumPy.

    Parameters
    ----------
    arr : numpy.ndarray
    win : int
        Number of samples in one window
    overlap : int
        Number of samples overlapping (0 means no overlap)
    pad : function
        padding function to be applied to data (if False
        will throw away data)
    axis : int
        Axis on which to apply the rolling window

    Returns
    -------
    shiftarr : ndarray
        Shifted copies of array segments

    Notes
    -----
    Using the `as_strided` function trick from Numpy means the returned
    array share the same memory buffer as the original array, so use
    with caution!
    Maybe `.copy()` the result if needed.
    This is the way for 2d array with overlap (i.e. step size != 1, which was the easy way):

    .. code-blocks::

        as_strided(a, (num_windows, win_size, n_features), (size_onelement * hop_size * n_feats, original_strides))
    """
    n_samples = len(arr)
    if not (win > 1 and win < n_samples):
        raise ValueError("window size must be greater than 1 and smaller than len(input)")
    if overlap < 0 or overlap > win:
        raise ValueError("Overlap size must be a positive integer smaller than window size")

    if not padding:
        raise NotImplementedError("As a workaround, please pad array beforehand...")

    if not _is_1d(arr):
        if axis is not 0:
            arr = np.swapaxes(arr, 0, axis)
        return chunk_data(arr, win, overlap, padding)

    return as_strided(arr, (win, n_samples - win + 1), (arr.itemsize, arr.itemsize))

def chunk_data(data, window_size, overlap_size=0, padding=False, win_as_samples=True):
    """Nd array version of :func:`shift_array`

    Notes
    -----
    Please note that we expect first dim as our axis on which to apply
    the rolling window.

    """
    assert data.ndim <= 2, "Data must be 2D at most!"
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows-1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # Or should I just NOT add this extra window?
    # The padding beforhand make it clear that we can handle edge values...
    if overhang != 0 and padding:
        num_windows += 1
        #newdata = np.zeros((num_windows * window_size - (num_windows-1) * overlap_size, data.shape[1]))
        #newdata[:data.shape[0]] = data
        #data = newdata
        data = np.pad(data, [(0, overhang+1), (0, 0)], mode='edge')

    size_item = data.dtype.itemsize

    if win_as_samples:
        ret = as_strided(data,
                         shape=(num_windows, window_size, data.shape[1]),
                         strides=(size_item * (window_size - overlap_size) * data.shape[1],) + data.strides)
    else:
        ret = as_strided(data,
                         shape=(window_size, num_windows, data.shape[1]),
                         strides=(data.strides[0], size_item * (window_size - overlap_size) * data.shape[1], data.strides[1]))

    return ret
    
def find_knee_point():
    "fin knee pount"
    pass

def mem_check(units='Gb'):
    "Get available RAM"
    stats = psutil.virtual_memory()
    units = units.lower()
    if units == 'gb':
        factor = 1./1024**3
    elif units == 'mb':
        factor = 1./1024**2
    elif units == 'kb':
        factor = 1./1024
    else:
        factor = 1.
        print("Did not get what unit you want, will memory return in bytes")
    return stats.available * factor
    