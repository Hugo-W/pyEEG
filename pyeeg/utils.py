# -*- coding: utf-8 -*-
"""
This module contains some utilities functions for data manipulation in general,
signal processing or/and text processing.
For instance function to create a matrix of lagged-time series for input to our
forward and backward models.

Also, utilities functions to apply *rolling* methods (rolling average, variance, or so on)
wiht the NumPy **side trick** see `the Python Cookbook recipee`_ for more details.

.. _the Python Cookbook recipee: https://ipython-books.github.io/47-implementing-an-efficient-rolling-average-algorithm-with-stride-tricks/
"""

#### Libraries
import logging
import psutil
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal as scisig
from scipy.fftpack import next_fast_len
#from numba import jit
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
# C-exensions functions
from pyeeg.ratemap import make_rate_map as ratemap
from pyeeg.ratemap import hz_to_erb_rate, erb_rate_to_hz, generate_cfs
from pyeeg.gammatone import gammatone_filter

logging.basicConfig(level=logging.ERROR)
LOGGER = logging.getLogger(__name__.split('.')[0])

def decorate_check_mne(func):
    """Decorator to check if MNE-Python is installed and import it if so.

    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    function
        Decorated function

    """
    def wrapper(*args, **kwargs):
        try:
            import mne
            return func(*args, **kwargs)
        except ImportError:
            raise ImportError(f"MNE-Python is not installed. Please install it to use this function ({func.__name__}).")
    return wrapper

def set_log_level(lvl):
    "Sets the log level globally across the PyEEG library."
    if isinstance(lvl, str): lvl = lvl.upper()
    logging.getLogger(__name__.split('.')[0]).setLevel(lvl)

def print_title(msg, line='=', frame=True):
    """Printing function, allowing to print a titled message (underlined or framded)

    Parameters
    ----------
    msg : str
        String of characters
    line : str
        Which character to use to underline (default "=")
    frame : bool
        Whether to frame or only underline title
    """
    print((line*len(msg)+"\n" if frame else "") + msg + '\n'+line*len(msg)+'\n')

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

    Raises
    ------
    ValueError
        If ``filling`` is set by user and ``drop_missing`` is ``True`` (it should be one or
        the other, the error is raised to avoid this confusion by users).

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
    if not np.isnan(filling) and drop_missing:
        raise ValueError("Dropping missing values or filling them are two mutually exclusive arguments!")

    dframe = pd.DataFrame(data)

    cols = []
    for lag in lag_samples:
        #cols.append(dframe.shift(-lag))
        cols.append(dframe.shift(lag))

    dframe = pd.concat(cols, axis=1)
    dframe.fillna(filling, inplace=True)
    if drop_missing:
        dframe.dropna(inplace=True)

    return dframe.values
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

def cochleogram(signal, srate, shift=8, nchannels=32, fmin=80, fmax=8000, comp_factor=1/3):
    """Compute the cochleogram of the input signal.

    Parameters
    ----------
    signal : ndarray (nsamples,)
        1-dimensional input signal
    srate : float
        Original sampling rate of the signal
    shift : float (default 8ms)
        Time shift between frames in ms
    nchannels : int (default 32)
        Number of channels in the filterbank
    fmin : float (default 80Hz)
        Minimum frequency of the filterbank
    fmax : float (default 8000Hz)
        Maximum frequency of the filterbank
    comp_factor : float (default 1/3)
        Compression factor (final envelope = env**comp_factor)

    Returns
    -------
    cochleogram : ndarray (nsamples, nchannels)
        Cochleogram
    """
    return ratemap(signal, srate, fmin, fmax, nchannels, shift, 8., comp_factor)
    

def signal_envelope(signal, srate, cutoff=20., method='hilbert', comp_factor=1./3, resample=125, verbose=None):
    """Compute the broadband envelope of the input signal.
    Several methods are available:

        - Hilbert -> abs -> low-pass (-> resample)
        - Rectify -> low-pass (-> resample)
        - subenvelopes -> sum

    The envelope can also be compressed by raising to a certain power factor.

    Parameters
    ----------
    signal : ndarray (nsamples,)
        1-dimensional input signal
    srate : float
        Original sampling rate of the signal
    cutoff : float (default 20Hz)
        Cutoff frequency (transition will be 10 Hz)
        In Hz
    method : str {'hilbert', 'rectify', 'subs'}
        Method to be used
    comp_factor : float (default 1/3)
        Compression factor (final envelope = env**comp_factor)
    resample : float (default 125Hz)
        New sampling rate of envelope (must be 2*cutoff < .. <= srate)
        Explicitly set to False or None to skip resampling

    Returns
    -------
    env : ndarray (nsamples_env,)
        Envelope
    
    """
    if verbose is None:
        verbose = 20 #  default to INFO?  or : LOGGER.getEffectiveLevel()
    if type(verbose) is str:
        verbose = logging._nameToLevel[verbose]
    LOGGER.log(verbose, "Computing envelope")

    if method.lower() == 'subs':
        return cochleogram(signal, srate, shift=1/resample*1000, nchannels=32, fmin=80, fmax=8000, comp_factor=comp_factor)
    else:
        if method.lower() == 'hilbert':
            # Get modulus of hilbert transform
            out = abs(scisig.hilbert(signal, N=next_fast_len(len(signal))))[:len(signal)]
        elif method.lower() == 'rectify':
            # Rectify signal
            out = abs(signal)
        else:
            raise ValueError("Method can only be 'hilbert', 'rectify' or 'subs'.")

        # Non linear compression before filtering to avoid NaN
        out = np.power(out + np.finfo(float).eps, comp_factor)
        # Design low-pass filter
        ntaps = fir_order(10, srate, ripples=1e-3)  # + 1 -> using odd ntaps for Type I filter,
                                                    # so I have an integer group delay (instead of half)
        b = scisig.firwin(ntaps, cutoff, fs=srate)
        # Filter with convolution
        out = scisig.convolve(np.pad(out, (len(b) // 2, len(b) // 2), mode='edge'),
                            b, mode='valid')
        #out = scisig.filtfilt(b, [1.0], signal) # This attenuates twice as much
        #out = scisig.lfilter(b, [1.0], pad(signal, (0, len(b)//2), mode=edge))[len(b)//2:]  # slower than scipy.signal.convolve method

        # Resample
        if resample:
            if not 2*cutoff < resample < srate:
                raise ValueError("Chose resampling rate more carefully, must be > %.1f Hz"%(cutoff))
            if srate//resample == srate/resample:
                env = scisig.resample_poly(out, 1, srate//resample)
            else:
                dur = (len(signal)-1)/srate
                new_n = int(np.ceil(resample * dur))
                env = scisig.resample(out, new_n)
    
    # Scale output between 0 and 1:
    return minmax_scale(env)

def fir_order(tbw, srate, atten=60, ripples=None):
    """Estimate FIR Type II filter order (order will be odd).

    If ripple is given will use rule:

    .. math ::

        N = \\frac{2}{3} \\log_{10}\\frac{1}{10\\delta_ripp\\delta_att} \\frac{Fs}{TBW}

    Else:

    .. math ::

        N = \\frac{Atten*Fs}{22*TBW} - 1

    Parameters
    ----------
    tbw : float
        Transition bandwidth in Hertz
    srate : float
        Sampling rate (Fs) in Hertz
    atten : float (default 60.0)
        Attenuation in StopBand in dB
    ripples : float (default None, optional)
        Maximum ripples height (in relative to peak)

    Returns
    -------
    order : int
        Filter order (i.e. 1+numtaps)

    Notes
    -----
    Rule of thumbs from here_.

    .. _here : https://dsp.stackexchange.com/a/31077/28372
    """
    if ripples:
        atten = 10**(-abs(atten)/10.)
        order = 2./3.*np.log10(1./10/ripples/atten) * srate / tbw
    else:
        order = (atten * srate) / (22. * tbw)
        
    order = int(order)
    # be sure to return odd order
    return order + (order%2-1)

def _is_1d(arr):
    "Short utility function to check if an array is vector-like"
    return np.prod(arr.shape) == max(arr.shape)

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

def rolling_func(func, data, winsize=2, overlap=1, padding=True):
    """Apply a function on a rolling window on the data
    """
    #TODO: check when Parallel()(delayed(func)(x) for x in rolled_array)
    # becomes advantageous, because for now it seemed actually slower...
    # (tesed only with np.cov on 15min of 64 EEG at 125Hx, list comprehension still faster)
    return [func(x) for x in chunk_data(data, win_as_samples=True, window_size=winsize, overlap_size=overlap, padding=True).swapaxes(1, 2)]

def moving_average(data, winsize=2):
    """#TODO: pad before calling chunk_data?
    """
    return chunk_data(data, window_size=winsize, overlap_size=(winsize-1)).mean(1)

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

    See Also
    --------
    :func:`pyeeg.utils.chunk_data`

    Notes
    -----
    Using the `as_strided` function trick from Numpy means the returned
    array share the same memory buffer as the original array, so use
    with caution!
    Maybe `.copy()` the result if needed.
    This is the way for 2d array with overlap (i.e. step size != 1, which was the easy way):

    .. code-block:: python

        as_strided(a, (num_windows, win_size, n_features), (size_onelement * hop_size * n_feats, original_strides))
    """
    n_samples = len(arr)
    if not (1 < win < n_samples):
        raise ValueError("window size must be greater than 1 and smaller than len(input)")
    if overlap < 0 or overlap > win:
        raise ValueError("Overlap size must be a positive integer smaller than window size")

    if padding:
        raise NotImplementedError("As a workaround, please pad array beforehand...")

    if not _is_1d(arr):
        if axis != 0:
            arr = np.swapaxes(arr, 0, axis)
        return chunk_data(arr, win, overlap, padding)

    return as_strided(arr, (win, n_samples - win + 1), (arr.itemsize, arr.itemsize))

def chunk_data(data, window_size, overlap_size=0, padding=False, win_as_samples=True):
    """Nd array version of :func:`shift_array`

    Notes
    -----
    Please note that we expect first dim as our axis on which to apply
    the rolling window.
    Calling :func:`mean(axis=0)` works if ``win_as_samples`` is set to ``False``,
    otherwise use :func:`mean(axis=1)`.

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

def find_knee_point(x, y, tol=0.95, plot=False):
    """Function to find elbow or knee point (minimum local curvature) in a curve.
    To do so we look at the angles between adjacent segments formed by triplet of
    points.

    Parameters
    ----------
    x : 1darray
        x- coordinate of the curve
    y : 1darray
        y- coordinate of the curve
    plot : bool (default: False)
        Whether to plot the result

    Returns
    -------
    float
        The x-value of the point of maximum curvature

    Notes
    -----
    The function only works well on smooth curves.
    """
    y = np.asarray(y).copy()
    y -= y.min()
    y /= y.max()
    coords = np.asarray([x,y]).T
    local_angle = lambda v1, v2: np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angles = []
    for k, coord in enumerate(coords[1:-1]):
        v1 = coords[k] - coord
        v2 = coords[k+2] - coord
        angles.append(local_angle(v1, v2))

    if plot:
        plt.plot(x[1:-1], minmax_scale(np.asarray(angles)/np.pi), marker='o')
        plt.hlines(tol, xmin=x[0], xmax=x[-1])
        plt.vlines(x[np.argmin(minmax_scale(np.asarray(angles)/np.pi)<=tol) + 1], ymin=0, ymax=1., linestyles='--')

    return x[np.argmin(minmax_scale(np.asarray(angles)/np.pi)<=tol) + 1]

def design_lagmatrix(x, nlags=1, time_axis=0):
    """
    Design a matrix of lagged time series.
    This is a helper function for autoregressive models estimation (so it will design a matrix of its own signal
    and will not use lag 0, only negative lag, i.e. to use for AR models and not ARMA).
    """
    if x.ndim == 1:
        time_axis = 1
    x = np.atleast_2d(x)
    if time_axis == 1: # transpose if time is in columns
        x = x.T
    n, k = x.shape # n: number of observations, k: number of dimensions
    X = np.zeros((n-nlags, nlags, k))
    for i in range(nlags):
        # X[:, i, :] = x[i+1:n-nlags+i+1]
        X[:, i, :] = np.roll(x, (i+1), axis=0)[nlags:, :]
    return X.squeeze(-1) if k==1 else X

def log_likelihood_lm(y, X, beta):
    """
    Compute the log likelihood of a linear model given the data and the parameters.

    Parameters
    ----------
    y : array_like
        The dependent variable. Shape (n, k), with n number of samples.
    X : array_like
        The independent variables. Shape (n, p), with p number of predictors.
    beta : array_like
        The parameters of the model. Shape (p, k), with k number of dependent variables.

    """
    n = len(y)
    residuals = y - X @ beta
    if y.ndim ==1:
        sigma2 = np.var(residuals, axis=0)
        log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(sigma2) - 1/(2*sigma2) * np.sum(residuals.T @ residuals, axis=0)
    else:
        sigma_hat = np.cov(residuals.T)  # Covariance matrix
        # Multivariate normal log-likelihood
        log_likelihood = -n/2 * np.log(np.linalg.det(sigma_hat)) - 1/2 * np.trace(np.linalg.inv(sigma_hat) @ (residuals.T @ residuals))
    return log_likelihood

def sigmoid(x, rmax=1, beta=1, x0=0):
    return rmax / (1 + np.exp(beta*(x0-x)))

def sigmoid_derivative(x, rmax=1, beta=1, x0=0):
    return beta * sigmoid(x, rmax, beta, x0) * (1 - sigmoid(x, rmax, beta, x0))

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
