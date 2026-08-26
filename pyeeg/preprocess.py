# -*- coding: utf-8 -*-
"""
Preprocessing helpers and functions to be applied to EEG data.

This module gathers covariance-based utilities and preprocessing transforms
commonly used in EEG/MEG analysis:

- Covariance estimators: ``covariance``, ``covariances``,
  ``covariances_extended`` (with wrappers around scikit-learn and NumPy
  covariance estimators in ``_check_est``).
- Filterbank helpers: ``create_filterbank``, ``apply_filterbank``, and
  ``get_power``.
- :class:`Whitener`: PCA/ZCA-based data whitening (demean, covariance,
  rotation, transform/inverse).
- :class:`WaveletTransform`: complex Morlet wavelet decomposition of a
  multi-channel signal.
- :class:`MultichanWienerFilter`: artifact removal with a multi-channel
  Wiener filter.

Functions applying a filterbank or wavelets accept an ``n_jobs`` argument to
parallelize the computation with joblib.
"""

#### Libraries
# Standard library
import numpy as np
from scipy import signal
from scipy.linalg import eigh as geigh
from joblib import Parallel, delayed

#import matplotlib.pyplot as plt
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from sklearn.base import TransformerMixin, BaseEstimator

# My libraries
from .utils import decorate_check_mne

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
    "Wrapper for numpy correlation estimator"
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
    """Estimation of covariance matrices from a list of "trials".
    
    Parameters
    ----------
    X : array-like (ntrials, nsamples, nchannels) or list
        If list, each element can also have different number of samples, but the number of
        channels must be the same for all trials.
    estimator : str
        One of covariance estimator from sklearn. See also :func:`_check_est`.

    Returns
    -------
    C : array-like (ntrials, nchannels, nchannels)
        The list of covariance matrices for each trial.
    """
    est = _check_est(estimator)
    if isinstance(X, list):
        Ntrials = len(X)
        Nchans = X[0].shape[1]
        assert all([x.shape[1] == Nchans for x in X]), "Inconsistent number of channels across trials."
    else: # here we assume X is then composed of similar length trials
        X = np.asarray(X)
        assert X.ndim == 3, "Data must be 3d (trials, samples, channels)"
        Ntrials, _, Nchans = X.shape
    covmats = np.zeros((Ntrials, Nchans, Nchans))
    for i in range(Ntrials):
        covmats[i, :, :] = est(X[i])
    return covmats

def covariance(X, estimator='cov'):
    """Estimation of one covariance matrix on the whole dataset. If X is of shape (trials, samples, channels)
    Will concatenate all trials together to compute a single covariance matrix across all of them.

    Parameters
    ----------
    X : ndarray (nsamples, nchannels) or (ntrials, nsamples, nchannels)
        Input data. If 3d, all trials are concatenated along the sample
        dimension before estimating the covariance.
    estimator : str or callable
        One of the covariance estimators understood by :func:`_check_est`
        (``'cov'``, ``'scm'``, ``'lwf'``, ``'oas'``, ``'mcd'``, ``'corr'``)
        or a callable returning a covariance matrix.

    Returns
    -------
    C : ndarray (nchannels, nchannels)
        The estimated covariance matrix.
    """
    est = _check_est(estimator)
    if X.ndim == 3:
        nchans = X.shape[2]
        X = X.reshape((-1, nchans))
    return est(X)


def covariances_extended(X, P, estimator='cov'):
    """Special form covariance matrix where data are appended with another set.
    For instance, the data could be EEG data and the other set could be a set of idealised response (e.g. a clean ERP).

    Parameters
    ----------
    X : ndarray (ntrials, nsamples, nchannels) or (nsamples, nchannels)
        Input data. If 2d, a dummy trial dimension is added (a single trial).
    P : ndarray (nsamples, nchannels_other)
        The other set appended to the data, e.g. an idealised response from
        the average across trials.
    estimator : str or callable
        One of the covariance estimators understood by :func:`_check_est`
        (``'cov'``, ``'scm'``, ``'lwf'``, ``'oas'``, ``'mcd'``, ``'corr'``)
        or a callable returning a covariance matrix.

    Returns
    -------
    C : ndarray (ntrials, nchannels + nchannels_other, nchannels + nchannels_other)
        The extended covariance matrix for each trial. If a single trial was
        given (2d ``X``), the leading dimension is squeezed out.

    Notes
    -----
    This assumes that the data are of shape (trials, samples, channels) and that the other set is of shape (samples, channels).
    The second set is typically an idealised response from the average across trials.
    The function could however also be called on a single trial, for continuous recordings for instance. In that case, the method used is
    to extend the data with the a dummy dimmension for the trials and for P convolve the idealise response to singular event with a series
    of impulses at the times of those events.

    """
    est = _check_est(estimator)
    if X.ndim == 2:
        X = X.reshape((1, X.shape[0], X.shape[1]))
    Ntrials, Nsamples, Nchans = X.shape
    Nsamples, Np = P.shape
    covmats = np.zeros((Ntrials, Nchans + Np, Nchans + Np))
    for i in range(Ntrials):
        covmats[i, :, :] = est(np.concatenate((P, X[i, :, :]), axis=0))
    return covmats.squeeze()

def create_filterbank(freqs, srate, filtertype=signal.cheby2, **kwargs):
    """Creates a filter bank, by default of chebychev type 2 filters.
    Parameters of filter are to be defined as name value pair arguments.
    Frequency bands are defined with boundaries instead of center frequencies.

    Parameters
    ----------
    freqs : list or ndarray of float
        Boundary frequencies of the bands (in Hz). Each value is normalized by
        the Nyquist frequency (``srate / 2``) and passed as the ``Wn`` argument
        to the filter design function.
    srate : float
        Sampling rate of the signal (in Hz).
    filtertype : callable
        Filter design function used to build each filter (e.g.
        ``scipy.signal.cheby2``, ``scipy.signal.butter``). Must accept ``Wn``
        and the keyword arguments in ``**kwargs``.
    **kwargs
        Additional name-value pairs passed to ``filtertype`` (e.g. ``Rs``,
        ``N``, or ``btype`` for Chebyshev type II filters).

    Returns
    -------
    fbank : list
        List of filter coefficients, one ``(b, a)`` tuple per frequency band.
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
        feat = np.array(Parallel(n_jobs=n_jobs)(delayed(signal.convolve)(signals[k, i, :]**2, signal.windows.boxcar(win)/win, 'same') for i in range(nchans)))
        if decibels:
            feat = 10 * np.log10(feat + 1e-16)
        out[k, :, :] = feat

    return out

class Whitener(TransformerMixin):
    """A data whitener (via either PCA or ZCA).

    Whitening linearly transforms the data so that its covariance becomes the
    identity matrix, decorrelating the channels and rescaling them to unit
    variance. Two standard whitening transforms are supported:

    - PCA whitening (``zca=False``): ``W = diag(1/sqrt(eigval)) @ eigvec.T``.
    - ZCA whitening (``zca=True``): ``W = eigvec @ diag(1/sqrt(eigval)) @ eigvec.T``,
      which additionally rotates the whitened data back to the original
      channel space (also called Mahalanobis or zero-phase whitening).

    The transform is ``(X - mu) @ W.T``; the mean ``mu`` is stored during
    :meth:`fit` (via :meth:`demean`) so that :meth:`transform` and
    :meth:`inverse` can be applied consistently to new data.

    Parameters
    ----------
    axis : int
        Axis along which to compute the mean and covariance. Default is 0
        (samples first).
    zca : bool
        If True, use ZCA whitening, otherwise PCA whitening (default False).
    bias : bool
        If True, divide by ``n`` when estimating the covariance; if False,
        use the unbiased estimator (divide by ``n - 1``). Default is True.

    Attributes
    ----------
    mu : ndarray or None
        Mean of the training data along ``axis``, computed by :meth:`demean`.
    sigma : ndarray or None
        Covariance matrix of the (demeaned) training data, computed by
        :meth:`cov`.
    W : ndarray or None
        The whitening matrix such that ``transform(X) = (X - mu) @ W.T``.
    scale : ndarray or None
        Diagonal matrix ``diag(1/sqrt(eigval))`` of inverse sqrt eigenvalues.
    U : ndarray or None
        Eigenvectors of the covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pyeeg.preprocess import Whitener
    >>> rng = np.random.default_rng(0)
    >>> M = np.array([[2., 0.5, 0.1], [0.5, 1., 0.3], [0.1, 0.3, 1.5]])
    >>> X = rng.standard_normal((100, 3)) @ M
    >>> wh = Whitener(axis=0, zca=True).fit(X)
    >>> Z = wh.transform(X)
    >>> np.allclose(np.cov(Z, rowvar=False), np.eye(3), atol=1e-1)
    True
    """
    def __init__(self, axis=0, zca=False, bias=True):
        """Initialize the Whitener.

        Parameters
        ----------
        axis : int
            Axis along which to compute the mean and covariance. Default is 0
            (samples first).
        zca : bool
            If True, use ZCA whitening, otherwise PCA whitening (default False).
        bias : bool
            If True, divide by ``n`` when estimating the covariance; if False,
            use the unbiased estimator (divide by ``n - 1``). Default is True.
        """
        self.zca = zca
        self.axis = axis
        self.bias = bias

        self.mu = None
        self.sigma = None
        self.W = None
        self.scale = None # 1/np.sqrt(eignval)
        self.U = None # eigenvectors

    def demean(self, data, axis=None):
        """Subtract the mean of ``data`` along ``axis`` and store it in ``self.mu``.

        Parameters
        ----------
        data : ndarray
            Input data.
        axis : int, optional
            Axis along which to compute the mean. If None, ``self.axis`` is
            used.

        Returns
        -------
        data_demeaned : ndarray
            ``data`` with the mean subtracted along ``axis``.
        """
        if axis is None: axis = self.axis
        self.mu = data.mean(axis=axis)
        return data - self.mu

    def cov(self, data, axis=None):
        """Estimate the covariance matrix of ``data`` and store it in ``self.sigma``.

        The covariance is computed as ``data.T @ data`` scaled by the number of
        samples (optionally debiased by one when ``self.bias`` is False).

        Parameters
        ----------
        data : ndarray
            Input data. Typically centered (see :meth:`demean`).
        axis : int, optional
            Axis along which samples run. If None, ``self.axis`` is used. The
            data are transposed so that samples run along axis 0.

        Returns
        -------
        sigma : ndarray (nchannels, nchannels)
            The estimated covariance matrix.
        """
        debias = 1. - int(self.bias)
        if axis is None: axis = self.axis
        if axis != 0: data = np.swapaxes(data, 0, axis)
        self.sigma = data .T @ data / (len(data) - debias)
        return self.sigma

    def compute_rotation(self, C=None):
        """Compute the whitening rotation matrix from the (stored) covariance.

        The eigenvalues/eigenvectors of the covariance matrix (either
        ``self.sigma`` or the provided ``C``) are used to build the whitening
        matrix ``W``. PCA whitening uses ``W = diag(1/sqrt(eigval)) @ eigvec.T``
        and ZCA whitening uses ``W = eigvec @ diag(1/sqrt(eigval)) @ eigvec.T``.

        Parameters
        ----------
        C : ndarray (nchannels, nchannels), optional
            Covariance matrix to diagonalize. If None, ``self.sigma`` (as
            computed by :meth:`cov`) is used.

        Returns
        -------
        None
            Stores ``self.scale``, ``self.U`` and ``self.W`` in place.

        Raises
        ------
        AssertionError
            If ``self.sigma`` has not been computed yet and ``C`` is None.
        """
        assert self.sigma is not None, "Compute covariance matrix first"
        if C is None:
            e, V = np.linalg.eigh(self.sigma)
        else:
            e, V = np.linalg.eigh(C)
        e = np.diag(1/np.sqrt(e))
        self.scale = e
        self.U = V
        if self.zca:
            self.W = V @ e @ V.T
        else:
            self.W = e @ V.T

    def fit(self, X, y=None, axis=None):
        """Fit the whitener on ``X``: demean, estimate the covariance and
        compute the whitening rotation.

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.
        axis : int, optional
            Axis along which samples run. If None, ``self.axis`` is used.

        Returns
        -------
        self : Whitener
            The fitted estimator.
        """
        self.cov(self.demean(X, axis=axis), axis=axis)
        self.compute_rotation()
        return self

    def transform(self, X, y=None):
        """Whiten ``X`` using the fitted mean and rotation.

        Parameters
        ----------
        X : ndarray
            Data to whiten. Must have the same number of channels as the
            training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        X_white : ndarray
            Whitened data, of the same shape as ``X``, with approximately
            identity covariance.
        """
        return (X - self.mu) @ self.W.T

    def fit_transform(self, X, y=None, axis=None):
        """Fit the whitener on ``X`` and return the whitened data.

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.
        axis : int, optional
            Axis along which samples run. If None, ``self.axis`` is used.

        Returns
        -------
        X_white : ndarray
            Whitened data, of the same shape as ``X``.
        """
        self.fit(X, axis=axis)
        return self.transform(X)

    def inverse(self, X, axis=None):
        """Invert the whitening transform, approximately recovering the
        original (unwhitened) data.

        The inverse uses the pseudo-inverse of ``W.T`` and adds back the
        stored mean ``self.mu``.

        Parameters
        ----------
        X : ndarray
            Whitened data.
        axis : int, optional
            Unused, kept for API compatibility.

        Returns
        -------
        X_orig : ndarray
            Data in the original channel space, of the same shape as ``X``.
        """
        return (X @ np.linalg.pinv(self.W.T)) + self.mu

@decorate_check_mne
class WaveletTransform(TransformerMixin):
    '''This class creates a list of wavelet transforms (complex Morlet wavelet)
    and applies it to a multi-channel signal.

    A bank of complex Morlet wavelets is built for the requested frequencies
    and each channel of the input signal is convolved with every wavelet,
    producing a time-frequency decomposition of the signal.

    Parameters
    ----------
    freqs : list or ndarray of float
        Center frequencies (in Hz) at which wavelets are created.
    sfreq : float
        Sampling frequency of the signal (in Hz).
    n_cycles : int or float
        Number of cycles of the Morlet wavelet (default 7). If an array of the
        same length as ``freqs``, each frequency gets its own number of cycles.

    Attributes
    ----------
    wavelets : list of ndarray
        The complex Morlet wavelets (one per frequency).
    nfreqs : int
        Number of frequencies (``len(freqs)``).
    sfreq : float
        Sampling frequency (in Hz).
    n_cycles : int or float
        Number of cycles used to build the wavelets.

    Requires MNE-Python to be installed (checked by :func:`decorate_check_mne`).
    '''
    def __init__(self, freqs, sfreq, n_cycles=7):
        """Create the wavelet bank with :func:`mne.time_frequency.morlet`.

        Parameters
        ----------
        freqs : list or ndarray of float
            Center frequencies (in Hz) at which wavelets are created.
        sfreq : float
            Sampling frequency of the signal (in Hz).
        n_cycles : int or float
            Number of cycles of the Morlet wavelet (default 7). If an array of
            the same length as ``freqs``, each frequency gets its own number
            of cycles.
        """
        from mne.time_frequency import morlet
        self.wavelets = morlet(sfreq, freqs, n_cycles=7)
        self.nfreqs = len(freqs)
        self.sfreq = sfreq
        self.n_cycles = n_cycles
        
    def transform(self, X, n_jobs=1):
        """Decompose a multi-channel signal with the wavelet bank.

        Parameters
        ----------
        X : ndarray (ntimes, nchannels)
        n_jobs : int
            Number of parallel jobs used to convolve the wavelets with each
            channel (via joblib, multiprocessing backend). Default is 1.

        Returns
        -------
        Y : ndarray (nfreqs, ntimes, nchannels)
            Complex time-frequency representation of the input signal.
        """
        Y = np.zeros((self.nfreqs, X.shape[0], X.shape[1]), dtype=np.complex64)
        for k in range(X.shape[1]):
            Y[..., k] = Parallel(backend='multiprocessing', n_jobs=n_jobs)(delayed(np.convolve)(X[:, k], w, 'same') for w in self.wavelets)
        return Y

class MultichanWienerFilter(BaseEstimator, TransformerMixin):
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

    Example of result obtained (cleaning EOG artifact here):
    
    .. image:: ../img/MWF_EOG_cleaning_example.png
        :width: 600
        
    '''
    def __init__(self, lags=(0,), low_rank=False, thresh=None):
        """Initialize the multichannel Wiener filter.

        Parameters
        ----------
        lags : tuple of int
            Lags used for the general model (NOT IMPLEMENTED YET). Default is
            ``(0,)``.
        low_rank : bool
            Whether to use a low-rank approximation of the covariance matrix
            of the artifactual data (default False).
        thresh : int or float, optional
            If int, corresponds to the rank prior on the artifact subspace.
            If float, considered as the percentage of variance to be kept.
            Only used when ``low_rank`` is True. If None, no thresholding is
            applied.
        """
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
        '''Filter the data to remove artifact learned by the model.

        Parameters
        ----------
        x : ndarray
            EEG data (samples, channels).

        Returns
        -------
        out : ndarray
            Filtered EEG data (artifacts removed), same shape as ``x``.

        Notes
        -----
        The method name ``transorm`` is a historical typo of ``transform`` but
        is retained as-is for backwards compatibility.
        '''
        return x - x @ self.W_
        
    def fit_transform(self, y_clean, y_artifact, x, cov_data=False):
        '''Train the model on input and transform directly the data in `x`.

        Parameters
        ----------
        y_clean : ndarray
            Clean segments used for training.
        y_artifact : ndarray
            Artifact-contaminated segments used for training.
        x : ndarray
            EEG data to filter.
        cov_data : bool
            Whether the input data are already covariance matrices estimated
            for each class (default False).

        Returns
        -------
        out : ndarray
            Filtered EEG data (artifacts removed), same shape as ``x``.
        '''
        self.fit(y_clean, y_artifact, cov_data=cov_data)
        return self.transorm(x)

