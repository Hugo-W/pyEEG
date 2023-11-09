# -*- coding: utf-8 -*-
"""
=====================
Connectivity measures
=====================

Available metrics:
    - WPLI (weighted phase lag index)
    - PLM (phase linearity measurement)

TODO:
    - PDC
    - Epoch based vs Continuous !?
    - Phase Transfer entropy!!! ()
"""
import numpy as np
from scipy.signal import hilbert, csd
from scipy.fftpack import fft, fftfreq

def phase_transfer_entropy(x, y, fs=1, nfft=None, fbands=None):
    """
    Compute phase transfer entropy between two signals x and y.
    """
    if nfft is None: nfft = x.shape[0]
    freqs = fftfreq(nfft, d=1/fs)
    if fbands is not None:
        fsel = np.where((freqs > fbands[0]) & (freqs < fbands[1]))
        freqs = freqs[fsel]
    else:
        fsel = slice(None)
    # compute phase of x and y
    phx = np.angle(hilbert(x, axis=0))
    phy = np.angle(hilbert(y, axis=0))
    # compute phase transfer entropy
    pte = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        # compute phase transfer entropy
        pte[i] = np.mean(np.abs(np.exp(1j*(phx[fsel, f] - phy[fsel, f]))))
    return pte

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
    with implemementation inspired by Java code by C.J. Stam (https://home.kpn.nl/stam7883/brainwave.html)
    Note that implementations differ in normalisation, as well as choices for binning and delay
    """
    assert len(np.atleast_2d(data).shape) == 2, "data must be a 2d array"
    if delay is not None: assert isinstance(delay, int), "delay must be an integer"
    N, chan = data.shape
    PTE = np.zeros((chan, chan)) # Phase Transfer Entropy matrix
    dPTE = np.zeros_like(PTE) # directed PTE

    cpx_data = hilbert(data, axis=0) # compute complex signal
    phi = np.angle(cpx_data) # compute phase (between -pi and pi)
    phi += np.pi # shift phase between 0 and 2pi

    # Compute delay if not provided
    if delay is None:
        # delay is based on the number of times the phase flips across time and channels, as in Brainwave (C.J. Stam)
        n1, n2 = 0, 0
        for i in range(chan):
            for j in range(i+1, chan):
                n1 += np.sum(np.abs(np.diff(np.sign(np.diff(phi[:, i])))))
                n2 += np.sum(np.abs(np.diff(np.sign(np.diff(phi[:, j])))))

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
    