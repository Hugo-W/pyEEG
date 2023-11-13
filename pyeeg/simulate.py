# -*- coding: utf-8 -*-
"""
=====================
Simulations utilities
=====================

Simulate MEEG-like signals with different connectivity patterns or methods.

TODO:
    - Simulation based on connectivity matrix
    - Neural mass models:
        - Jansen-Rit -> See https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00973/1-s2.0-S1053811903006566/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJHMEUCIEBRfmnMkb2j8ut2p46cc2y6emk9Jl4srBYEgaVRmPS%2FAiEAtKpadf9qZgbW0vxfxuG0FxR2CXU8iGh9M9VZ6T36r%2BMqswUISBAFGgwwNTkwMDM1NDY4NjUiDAOu3ZZjVACJ4G%2B%2F4yqQBbt8wNEj1LK7ozKjn8OBKbG3gcdZaU0Sg%2FLJMMQfgZnBK0iFT%2BHicPKc%2BEzpvhyikXNHGv%2FXtokrD%2Favx5YbL%2B7rKox9FRVvl9pVFrdaSkf%2BhkE2ACQM6nxOFXVbfpsd0QSEGYH40O8EL%2F5FQOxBcuLU2SE1wA9xSjpHnQB0CM1Q2WxF67v0WAkiSIuwAA5hwfscGL%2BUzRidKmTWC8B8lzPKki0D0jZVngPHRHuawuZbmR07LHye1pgPBacqY%2B3DBrOrIjgZXU%2FMzPw1kgcou%2Fd0nJnJgfEFotmAuhj%2FgbLPuOu0ROhYeJeCUTFQ9cXwaIkN%2FpGPwV3EtLvkH7QvEpsfKDMOR1iB4YYMh1oe75O7lx6hJ5qWpHCFgEh0Xg5Z%2BatTlrifNzMyr5isVLVrUZElzdY%2B3ZCZKaX%2FA6zJBp0Y%2FsDCvV4IU5K2GHOzTWyimbTQd2WUk%2BM4FEpn9U38zsTnEA5I5%2F5il8dMHS4fRDCF0dEarsdRbi%2BRXuLnFU8%2FTkkRyMGeqh4nIt8%2F7MyvrWE5D2YqSwImBeknhAVQ%2BF%2BHUWqU9tNVrnbuchks%2BZWB4E2%2BkEJeUBtFlswq5W5DuPMVAzT0IbCkITPSLvHtyAuAvPD2S3zfduMgPu98lwYfijcDEdS58uc05WQiUIHg8H3u%2F46DHT8MbHFp7B8FgkWqYWzof1yCIA6dOzjD9AvkSjGCxtbUqiDgGV88pzvnS9CKEX%2BKt1k5bLebz6MERTnPBd%2BTbrFGpHJlzKB%2FCbfVIkEzqvZQIKCkMUm7I8JfainxdL3y4JjKMxi75aoenMZSPOhBRPNQFglT43ZYjuZXNitc4P1KpEDkIGkS7iSdeU5576MRicX3fID9feeeMGlYMPf8yKoGOrEBRrcOQuw9PMvlCvY0kUumTwNahcqJuAbPVCj5UFZKOf9bBMEOcC1Xbi5r%2FCoOYPPf9yyQIuQ4pk5UTOs3E%2BBHrMqCboKDNcLFdkc6oWOtD8yF%2BaZddbsrfT2%2FdIWe80mM65qPFbm374EtTfoo8mGY%2F8QwlzJfEUYFSUgpZr5%2B32O2pJNnNIGQxL3suTL9hEGSvZAaxEgE3TcP073vpVxDHZzWPai90pW%2FtrJCH0v9vkdi&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231113T164609Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRX55LHJ7%2F20231113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=efb4397d32d0283d891f7f7b6b54cf6086f166c7f49d7c73cd85383c2ace1215&hash=c12ba03b2ecaf8322bf34c85889ef48d9263bd5c22c8fe4fdc4fd77c9ddd950c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903006566&tid=spdf-fd17f1b2-9263-4484-94b2-caa5ff6aeb65&sid=1f34fbf92ab4c94b979becc2279c4fddfea3gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f56555356560150&rr=82587d3bad7f66a5&cc=nl
        or https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X00924/1-s2.0-S1053811903004579/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAAaCXVzLWVhc3QtMSJIMEYCIQCLOeRLcqnzm1kuLuu9gAR4xW20REgfwC%2BUG7p0FPu14gIhANRW6LXlFS2LON%2F%2BgS1jbn%2F7a8w6IHV7gKbTbNV5wqjnKrIFCEkQBRoMMDU5MDAzNTQ2ODY1Igy1V1bpLECYjeUFuUEqjwVSOS4pH6j2ew1gfsyDFISqu%2FmCuyDnXJ%2BkanuTqSOt35WOcurB7Pczd0RdZV4dTzoTbW0ZoPMv6dCNi8tfQ4oaeaxWA3mN0Vs%2FIaYMy8eyCOkgxTFBPEOxGmXFV1hBWQhmske05vPkhjTf37BfdI7Ipik7o%2BHMGfIKoQKO8zybRghfnsSyV3crhDAbQx9GF%2BOe%2FKNp7X5Yx06OV953HqhGiu%2Fs5wITv2fSRaAvoLHoMkiYLo8B0jNIynMY71cB9p68YgUPZ%2FAxfUiSHT%2FwjTywZO1sPhVSB0Bm2HqCDh5soRiZ07MylsEDbYbnrW%2B7aCVBRKykjaQ9IFK4OwiaE%2BK7WQm6pdztg3oLmY5Su%2FWU90iUY41Ju1dUGmhsFfO8Q8WTlnnZ5GBRzGcgqKxuYcKHWxUBieD5U6w2FSynBti2ryxOFiiGtXI%2BeODXN1Ea0qoR9cT7HLExDbFyL101aLiyP%2FTSFjq%2Buygyfwl0yk6%2FkJdgWZYB3Xllmi9s7uc3sZ3KsZo5P4m8we2fz5XrMDpCLsiuTMfUwfZatWon8GXyysOSOHwoCVvS%2Btn%2BnZxRGbNTcNvvRmSQz5TGJ02lKzNOC5Bd0y9fKCfuNSXVcL1FutS2oJtonpy0iiLc3mgczpXspnK3kJaSAJo%2FFXaW8OVupYm3C8hQx5E9wCb1c%2FdSCLcaitk%2FZA3eGjbIXjcUiTHuiit6N20F6VhzUNTkSuM7FZzNH%2Fkl7Cmt72Z13TDiG%2BUkrD3RA2dePMS5XvqkwiWvLWK%2Bbljweot6RbbwfRoLc4ouc8Fy4pVsUhlABP6iAFGRpz7lGEUDo6cxpYp1J8aoUbpnibvPR1SdHUSz%2BGbq7PXxmjEO%2Bu0Lu2D7JZqAMO2LyaoGOrABbIrfm40nIMybDuTZ4mK%2BoaIvNb9Vc0xLy1Nypl6sNZoLJl4oVqI0WYNwtHXNhpbGBR8jq2%2F%2FL24tih289cLG8S5GV07PE5oeIJj7DfywGo8PPko2O6vViv9%2BZo4i5RIEn259RUd2IXplHZd57rggnFTY4ac8m9CEbjtk7gTW7HF7DNFaFMRv6D8K%2F5hpLKGHjyF%2BZhxZ79fcAN%2B5TgvVIBYRWS4Qi%2F%2B6NMhnU28QTVo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231113T164603Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY742FACEQ%2F20231113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=da2d7fe1b02d5c068a1f3c7c8f2fdc64b8350e1f909000d3d6906267796778a2&hash=82bcc505cc92fe4f322349938bc58f4add1575143448707ba5b16def04ceade7&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811903004579&tid=spdf-2c8c5161-288d-41b8-83ac-547a9870612e&sid=1f34fbf92ab4c94b979becc2279c4fddfea3gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080f56555356560152&rr=82587d1569b866a5&cc=nl
        - Wilson-Cowan
        - Wong-Wang
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