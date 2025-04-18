import numpy as np
from functools import reduce
from tqdm import tqdm
import logging
from typing import Union, List

LOGGER = logging.getLogger(__name__)

def _lstsq_regress(x: Union[np.ndarray, List[np.ndarray]],
                  y: Union[np.ndarray, List[np.ndarray]],
                  verbose: bool = False) -> np.ndarray:
    """Linear regression using lstsq.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats) or list of such
        If a list of such is given (with possibly different nsamples), covariance matrices
        will be computed by accumulating them for each trials. The number of samples must then be the same
        in both x and y per each trial.
    y : ndarray (nsamples, nchans) or list of such
        If a list of such arrays is given, each element of the
        list is treated as an individual subject, the resulting `betas` coefficients
        are thus computed on the averaged covariance matrices.

    Returns
    -------
    betas : ndarray (nfeats, nchans)
        Coefficients

    Raises
    ------
    AssertionError
        If trial length for each x and y differ.

    Notes
    -----
    A warning is shown in the case where nfeats > nsamples, if so the user
    should rather use partial regression.
    """
    if not isinstance(x, list) and np.ndim(x) == 2:
        if x.shape[0] < x.shape[1]:
            LOGGER.warning("Less samples than features! The linear problem is not stable in that form. Consider using partial regression instead.")

    if (len(x) == len(y)) and np.ndim(x[0])==2: # will accumulate covariances
        assert all([xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(x, y)]), "Inconsistent trial lengths!"
        XtX = reduce(lambda x, y: x + y, [xx.T @ xx for xx in x])
        XtY = np.zeros((XtX.shape[0], y[0].shape[1]), dtype=y[0].dtype)
        count = 1
        if verbose:
            pbar = tqdm(total=len(x), leave=False, desc='Covariance accumulation')
        for X, Y in zip(x, y):
            if verbose:
                LOGGER.info("Accumulating segment %d/%d", count, len(x))
                pbar.update()
            XtY += X.T @ Y
            count += 1
        if verbose: pbar.close()
        
        betas = np.linalg.lstsq(XtX, XtY)[0]
    else:
        betas = np.linalg.lstsq(x, y)[0]
    return betas

def _svd_regress(x: Union[np.ndarray, List[np.ndarray]],
                 y: Union[np.ndarray, List[np.ndarray]],
                 alpha: Union[float, np.ndarray],
                 verbose: bool = False) -> np.ndarray:
    """Linear regression using svd.

    Parameters
    ----------
    x : ndarray (nsamples, nfeats) or list of such
        If a list of such is given (with possibly different nsamples), covariance matrices
        will be computed by accumulating them for each trials. The number of samples must then be the same
        in both x and y per each trial.
    y : ndarray (nsamples, nchans) or list of such
        If a list of such arrays is given, each element of the
        list is treated as an individual subject, the resulting `betas` coefficients
        are thus computed on the averaged covariance matrices.
    alpha : float or array-like
        If array, will compute betas for every regularisation parameters at once

    Returns
    -------
    betas : ndarray (nfeats, nchans, len(alpha))
        Coefficients

    Raises
    ------
    ValueError
        If alpha < 0 (coefficient of L2 - regularization)
    AssertionError
        If trial length for each x and y differ.

    Notes
    -----
    A warning is shown in the case where nfeats > nsamples, if so the user
    should rather use partial regression.
    """
    # cast alpha in ndarray
    if np.isscalar(alpha):
        alpha = np.asarray([alpha], dtype=float)
    else:
        alpha = np.asarray(alpha)

    if not isinstance(x, list) and np.ndim(x) == 2:
        if x.shape[0] < x.shape[1]:
            LOGGER.warning("Less samples than features! The linear problem is not stable in that form. Consider using partial regression instead.")

    try:
        assert np.all(alpha >= 0), "Alpha must be positive"
    except AssertionError:
        raise ValueError

    if (len(x) == len(y)) and np.ndim(x[0])==2: # will accumulate covariances
        assert all([xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(x, y)]), "Inconsistent trial lengths!"
        XtX = reduce(lambda x, y: x + y, [xx.T @ xx for xx in x])
        [U, s, V] = np.linalg.svd(XtX, full_matrices=False) # here V = U.T
        XtY = np.zeros((XtX.shape[0], y[0].shape[1]), dtype=y[0].dtype)
        count = 1
        if verbose:
            pbar = tqdm(total=len(x), leave=False, desc='Covariance accumulation')
        for X, Y in zip(x, y):
            if verbose:
                LOGGER.info("Accumulating segment %d/%d", count, len(x))
                pbar.update()
            XtY += X.T @ Y
            count += 1
        if verbose: pbar.close()
        # XtY /= len(x) # NO: IT SHOULD BE A SUM
        
        #betas = U @ np.diag(1/(s + alpha)) @ U.T @ XtY
        
        eigenvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        eigenvals_scaled[range(len(V)), range(len(V)), :] = 1 / \
            (np.repeat(s[:, None], np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        Vsreg = np.dot(V.T, eigenvals_scaled) # np.diag(1/(s + alpha))
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, U.T @ XtY) #Vsreg @ Ut
    else:
        [U, s, V] = np.linalg.svd(x, full_matrices=False)
        if np.ndim(y) == 3:
            Uty = np.zeros((U.shape[1], y.shape[2]))
            for Y in y:
                Uty += U.T @ Y
            Uty /= len(y)
        else:
            Uty = U.T @ y

        # Broadcast all alphas (regularization param) in a 3D matrix,
        # each slice being a diagonal matrix of s/(s**2+lambda)
        eigenvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        eigenvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(s[:, None], np.size(alpha), axis=1) / \
            (np.repeat(s[:, None]**2, np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        # A dot product instead of matmul allows to repeat multiplication alike across third dimension (alphas)
        Vsreg = np.dot(V.T, eigenvals_scaled) # np.diag(s/(s**2 + alpha))
        # Using einsum to control which access get multiplied, again leaving alpha's dimension "untouched"
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty) #Vsreg @ Uty
    
    return betas
