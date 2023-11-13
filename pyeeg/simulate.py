# -*- coding: utf-8 -*-
"""
=====================
Simulations utilities
=====================

Simulate MEEG-like signals with different connectivity patterns or methods.

TODO:
    - Simulation based on connectivity matrix
    - Neural mass models
    - covariance-based simulation to be checked!!!

Update:
    - 10/11/2023: initial commit
"""
import numpy as np

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

    return data[order:, :]

def simulate_var_from_cov(cov, nobs=500, ndim=2, seed=42, verbose=False):
    """
    Simulate a VAR model of order `order` from a covariance matrix.

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
    order = cov.shape[0]
    assert cov.shape == (order, ndim, ndim), "cov must be of shape (order, ndim, ndim)"
    data = np.zeros((nobs+order, ndim))

    if verbose:
        print(f"Simulating VAR({order}) model with {ndim} dimensions and {nobs} observations")
        print(f"Data shape: {data.shape}")

    data[:, :] = rng.standard_normal(size=data.shape) # initialize with noise
    for t in range(order, nobs+order):
        for lag in range(order):
            data[t] += data[t-(lag+1)] @ np.linalg.cholesky(cov[lag]) # here if I multiply from the left, I get the contributions row wise instead of column wise

    return data[order:, :]