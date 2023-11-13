# -*- coding: utf-8 -*-
"""
=====================
Connectivity measures
=====================

Available metrics:
    - WPLI (weighted phase lag index)
    - PLM (phase linearity measurement)
    - phase transfer entropy (PTE)

Some utilities:
    - jackknife resampling
    - cross-spectral density (CSD)
    - simulation of VAR model signals

TODO:
    - PDC
    - Epoch based vs Continuous !?
    - Coherence
    - Imaginary Coherence
    - Phase Coherence
    - SL (Synchronization Likelihood)
    - PLT (Phase Lag Time)
    - AEC (amplitude envelope correlation)

Update: 
- 2021/03/11: Added phase_transfer_entropy, simulate_ar, simulate_var
"""
import numpy as np
from scipy.signal import hilbert, csd
from scipy.fftpack import fft, fftfreq


def pte_draft(data, delay=None, binsize='scott'):
    """
    Compute Phase Transfer Entropy between each pair of channels in data.

    ---
    Original MATLAB code from Matteo Fraschini, Arjan Hillebrand (VERSION 2.5 /2017)
    accessed at: https://figshare.com/articles/Phase_Transfer_Entropy/3847086
    Cite:
    Fraschini, Matteo; Hillebrand, Arjan (2017). Phase Transfer Entropy in Matlab. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3847086.v12

    Following description of PTE from:
    M Lobier, F Siebenhuhner, S Palva, JM Palva (2014) Phase transfer entropy: a novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions. Neuroimage 85, 853-872
    with implemementation inspired by Java code by C.J. Stam (https://web.archive.org/web/20200711091249/https://home.kpn.nl/stam7883/brainwave.html)
    Note that implementations differ in normalisation, as well as choices for binning and delay
    """
    assert len(np.atleast_2d(data).shape) == 2, "data must be a 2d array"
    if delay is not None: assert isinstance(delay, int), "delay must be an integer"
    N, chan = data.shape

    # Compute phase
    phi = np.angle(hilbert(data, axis=0))

    # Compute ideal delay
    count1, count2 = 0, 0
    for j in range(chan):
        for i in range(1, N-1):
            count1 += 1
            if (phi[i-1, j] * phi[i+1, j]) < 0:
                count2 += 1
    print(np.round(count1/count2))
    delay = int(np.round(count1/count2))

    phi += np.pi # get it between 0 and 2pi

    # Binsize
    method = 'scott'
    if method=='scott':
        binsize = 3.49*np.std(phi, 0).mean()/np.power(N, 1/3) # binsize as in Scott et al. 2014
    elif method=='fd':
        binsize = 2*(np.percentile(phi, 75) - np.percentile(phi, 25))*np.power(N, -1/3) # binsize as in Freedman et al. 1981
    elif method=='otnes':
        # binsize based on Otnes R. and Enochson (1972) Digital Time Series Analysis. Wiley.
        # as in Brainwave (C.J. Stam)
        # Nbins = np.round(np.log2(N)) + 1
        Nbins = np.exp(0.626 + 0.4*np.log(N-(delay+1)))
        binsize = 2*np.pi/Nbins

    # get the bins
    bins_w = np.arange(0, 2*np.pi, binsize)
    bins_w = np.r_[np.arange(0, 2*np.pi, binsize), 2*np.pi] # add the last bin
    Nbins = len(bins_w) - 1
    
    PTE = np.zeros((chan, chan))
    for i in range(chan):
        for j in range(chan):
            # Compute phase transfer entropy between i and j: namely x -> y (x causes y)
            if i==j: continue
            # Using numpy's histogram is much faster and gives the same result
            # ypr stands for "y predicted", i.e. ahead from y (such that P(ypr|y) is the probability of y given y past)
            Py = np.histogram(phi[:-delay, j], bins_w)[0].astype(float)
            Py_x = np.histogram2d(phi[:-delay, j], phi[:-delay, i], bins_w)[0].astype(float)
            Pypr_y = np.histogram2d(phi[delay:, j], phi[:-delay, j], bins_w)[0].astype(float) # conditioned on y past
            # And conditionned on both x and y past:
            Pypr_yx, edges = np.histogramdd(np.c_[phi[delay:, j], phi[:-delay, j], phi[:-delay, i]], bins=(bins_w, bins_w, bins_w))
            Pypr_yx = Pypr_yx.astype(float)

            # Normalise probabilities:
            Py /= N - delay
            Py_x /= N - delay
            Pypr_y /= N - delay
            Pypr_yx /= N - delay

            # Compute entropies
            Hy = -np.nansum(Py*np.log2(Py))
            Hy_x = -np.nansum(Py_x*np.log2(Py_x))
            Hypr_y = -np.nansum(Pypr_y*np.log2(Pypr_y))
            Hypr_yx = -np.nansum(Pypr_yx*np.log2(Pypr_yx))

            # Compute PTE
            PTE[i, j] = (Hy - Hy_x - Hypr_y + Hypr_yx)/np.log2(Nbins) # normalising by log2(Nbins) as in Brainwave (C.J. Stam) ?
            # matlab version as its opposite:
            # PTE[i, j] = Hypr_y + Hy_x - Hy - Hypr_yx # and no normalisation

    # Compute dPTE
    tmp = np.triu(PTE) + np.tril(PTE).T
    dPTE = np.tril(PTE/tmp.T, -1) + np.triu(PTE/tmp, 1)
    return dPTE, PTE


def jackknife_resample(data):
    """
    Performs jackknife resampling on numpy array.

    This technique generates `n` samples of length `n-1` from the input of length `n`.
    The ith sample has the ith measurment "knocked-out".
    This statistical method may be useful in estimating covariance-based metrics from single
    trial data.
    """
    n = data.shape[0]
    resampled = np.zeros((n, n-1, *data.shape[1:]))

    for i in range(n):
        resampled[i] = np.delete(data, i, axis=0)

    return resampled

def csd_ndarray(x, fs=1, nfft=None):
    """Compute the cross-spectral density between each column of x.
    """
    N, nchans = x.shape
    if nfft is None: nfft = N

    S = np.zeros((nchans, nchans, nfft//2+1), dtype=np.complex)
    for i in range(nchans):
        for j in range(i, nchans):
            c = csd(x[:, i], x[:, j], fs=fs, nfft=nfft, nperseg=N)[1]
            S[i, j] = c
            S[j, i] = c
    return S

def simulate_ar(order, coefs, n, sigma=1, seed=42):
    """
    Simulate an autoregressive process of order `order`.

    Parameters
    ----------
    order : int
        The order of the autoregressive process.
    coefs : array_like
        The coefficients of the autoregressive process. The first element is the coefficient of the lag (t-1).
    n : int
        The number of samples to simulate.
    sigma : float
        The standard deviation of the additive noise process.
    
    Returns
    -------
    x : array_like
        The simulated time series. Shape (n,).
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n + order)
    for i in range(n+order):
        if i < order:
            x[i] = rng.standard_normal()* sigma
        else:
            x[i] = np.dot(coefs[::-1], x[i-order:i]) + rng.standard_normal()*sigma
    return x[order:]


def simulate_var(order, coef, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order`.

    The VAR model is defined as:

    .. math::
        x_t = A_1 x_{t-1} + A_2 x_{t-2} + ... + A_p x_{t-p} + \\epsilon_t
        x_t = \\sum_{i=1}^p A_i x_{t-i} + \\epsilon_t

    where :math:`x_t` is a vector of shape (ndim, 1), :math:`A_i` is a matrix of shape (ndim, ndim)

    .. note::
        The coefficients at a given lag are such as :math:`C_ij` is i->j, so it will be the coefficients for dimension j!
        For example, each row of the first column are determining the contributions of each component onto the first component.
    """
    rng = np.random.default_rng(seed)
    if order == 1 and coef.ndim == 2:
        coef = coef[None, :, :]
    assert coef.shape == (order, ndim, ndim), "coef must be of shape (order, ndim, ndim)"
    data = np.zeros((nobs+order, ndim))

    if verbose:
        print(f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations")
        print(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape) # initialize with noise
    for t in range(order, nobs+order):
        for lag in range(order):
            data[t] += data[t-(lag+1)] @ coef[lag] # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :] # remove the first order samples

def wPLI(x, fs=1, nfft=None, fbands=None):
    # see https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_wpli.m for variance estimation of jackknife
    # TODO:
    # bias = 1
    # if x.ndim < 3:
    #     x = jackknife_resample(x)
    #     bias = (n-1)**2
    if x.ndim < 3:
        x = jackknife_resample(x)
    Ntr, nsamples, nchan = x.shape
    freqs = fftfreq(nsamples, d=1/fs)
    num = np.zeros((nchan, nchan, len(freqs)//2 + 1))
    denom = np.zeros_like(num)
    # Summing across "trials"
    for k in range(Ntr):
        S = csd_ndarray(x[k], fs=fs) # compute CSD (nchan, nchan, nfreq)
        num += np.abs(S.imag) * np.sign(S.imag) # numerator
        denom += np.abs(S.imag) # denominator (normalisation)
    # wPLI:
    C = num/denom

    if fbands is not None:
        # Average over frequency range
        fsel = np.where((freqs > fbands[0]) & (freqs < fbands[1]))
        return C[..., fsel].mean(-1).squeeze()
    else:
        # return full wPLI across all freqs
        return C
    

def plm(x, fband=1, fs=1, rowvar=False):
    """
    Compute Phase Linearity Measurement (see [`1`_]).

    Parameters
    ----------
    x : ndarray shape (nsamples, nchannels)
    fband: float
        Frequency of interest (bandwidth on which to integrate)
        FFT of the interferometric signal (hilbert of x times hilbert y*).
    rowvar : bool
        This can be set to true if nchannels is the first dimension.

    .. _1: https://pubmed.ncbi.nlm.nih.gov/30403622/
    """
    assert fband < fs/2, "fband must be smaller than nyquist frquency."
    if rowvar:
        x = x.T
    N, nchan = x.shape
    xh = hilbert(x, axis=0)
    freqs = fftfreq(N, d=1/fs)
    C = np.zeros((nchan, nchan))
    bandwidth = abs(freqs) < fband
    for i in range(nchan):
        for j in range(i+1, nchan):
            z = xh[:, i] * np.conj(xh[:, j]) # compute interferometric signal
            z /= np.abs(z) # normalise
            z[0] = z[0] * (abs(np.angle(z[0])) > 0.1) # removing volume conduction
            Sz = np.abs(fft(z)**2) # compute Spectral density of z
            C[i, j] = Sz[bandwidth].sum()/Sz.sum()
            C[j, i] = Sz[bandwidth].sum()/Sz.sum() # undirectionnal, symmetrical measure

    return C
    