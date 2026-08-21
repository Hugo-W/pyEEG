# pylint: disable=invalid-name
"""
VAR model estimation helpers.

This module provides functions for fitting autoregressive (AR) and
vector autoregressive (VAR) models to (multivariate) time series.
"""

import numpy as np

from ..utils import design_lagmatrix


def fit_ar(x, nlags=1, time_axis=0):
    """
    Fit an autoregressive model to a time series.
    This is a helper function for autoregressive models estimation.

    If fed with a multidimensional time series, it will fit a model for each
    dimension.

    Parameters
    ----------
    x : ndarray (nsamples, nchans) or (nsamples, )
        Time series to fit.
    nlags : int
        Number of lags to use in the model (model order).
    time_axis : int
        Axis of the time series.

    Returns
    -------
    betas : ndarray (nchans, nlags)
        Coefficients of the autoregressive model.
    """
    if x.ndim == 1:
        time_axis = 1
    x = np.atleast_2d(x)
    if time_axis == 1:  # transpose if time is in columns
        x = x.T
    n, k = x.shape  # n: number of observations, k: number of dimensions

    X = design_lagmatrix(
        x, nlags=nlags, time_axis=0
    )  # time axis was already transposed
    if k == 1:
        X = np.atleast_3d(X)
    Y = x[nlags:, :]

    betas = np.zeros((k, nlags))
    for i in range(k):
        betas[i] = np.linalg.lstsq(X[:, :, i], Y[:, i], rcond=None)[0]
    return betas.squeeze() if k == 1 else betas


def fit_var(x, nlags=1, time_axis=0):
    """
    Fit a VAR model to a time series.

    Instead of fitting k independent models as in `fit_ar`, this function fits a single model
    but with multivariate regressors.

    Parameters
    ----------
    x : ndarray (nsamples, nchans) or (nsamples, )
        Time series to fit.
    nlags : int
        Number of lags to use in the model (model order).
    time_axis : int
        Axis of the time series.

    Returns
    -------
    betas : ndarray (nchans, nlags)
        Coefficients of the autoregressive model.
    """
    if x.ndim == 1:
        time_axis = 1
    x = np.atleast_2d(x)
    if time_axis == 1:  # transpose if time is in columns
        x = x.T
    n, k = x.shape  # n: number of observations, k: number of dimensions

    X = design_lagmatrix(
        x, nlags=nlags, time_axis=0
    )  # time axis was already transposed
    if k == 1:
        X = np.atleast_3d(X)
    Y = x[nlags:, :]
    # Now instead of looping over thrid axies of X (dimensions), we reshape it to a 2D matrix
    # And fit a single model
    betas = np.linalg.lstsq(X.reshape(-1, nlags * k), Y, rcond=None)[0]
    return betas.reshape(k, nlags, k)  # .transpose(2, 1, 0) # reshape back to 3D