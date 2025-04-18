import numpy as np
from functools import reduce
from tqdm import tqdm
import logging
from typing import Union, List

LOGGER = logging.getLogger(__name__)

def svd_solver(A, b, lambda_=0., truncated_svd=False, verbose=False):
    """
    Solve the linear system Ax = b using the SVD method.
    
    This method assunes that we are solving the normal equation:
    (X^T X + lambda I) x = X^T y
    Thus, A = X^T X and b = X^T y.

    Parameters:
    A : ndarray
        Matrix A. Typically of shape (n_features * n_lags, n_features * n_lags) in the context of TRF.
    b : ndarray
        Right-hand side vector. Typically of shape (n_features * n_lags, n_outputs) in the context of TRF.
    lambda_ : float, optional
        Regularization parameter.
    truncated_svd : bool, optional
        Whether to use the truncated SVD method. If True, lambda_ must be between 0 and 1; 
        it represents the fraction of the total variance to keep.

    Returns:
    x : ndarray
        Solution vector.
    """
    # Check symmetricity of A
    assert np.allclose(A, A.T), 'Matrix A must be symmetric'
    U, s, Vt = np.linalg.svd(A, full_matrices=False, hermitian=True)
    if truncated_svd:
        assert 0 < lambda_ < 1
        n_components = np.sum(np.cumsum(s) / np.sum(s) < lambda_) + 1

        if verbose: 
            print(f'Keeping {n_components} components (out of {len(s)})')
            print(f'Variance explained: {s[:n_components].sum() / s.sum()}')
            print(f"Singular values: {s[:n_components]}")
        U = U[:, :n_components]
        s = s[:n_components]
        Vt = Vt[:n_components, :]
        lambda_ = 0.
    s_inv = np.diag(1 / (s + lambda_))
    return Vt.T @ s_inv @ U.T @ b


def incomplete_cholesky_preconditioner(A):
    """
    Compute the Incomplete Cholesky preconditioner for matrix A.

    Parameters:
    A : ndarray
        Symmetric positive-definite matrix.

    Returns:
    M_inv : function
        Function that applies the preconditioner.
    """
    A_sparse = csc_matrix(A)
    ilu = spilu(A_sparse)
    M_inv = lambda x: ilu.solve(x)
    return M_inv

def diagonal_preconditioner(A):
    """
    Compute the Diagonal preconditioner for matrix A.

    Parameters:
    A : ndarray
        Symmetric positive-definite matrix.

    Returns:
    M_inv : function
        Function that applies the preconditioner.
    """
    diag = np.diag(A)
    M_inv = lambda x: x / diag
    return M_inv

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None, lambda_=0., preconditioner=None, verbose=False):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method. A must be square, symmetric and positive-definite.

    Parameters:
    A : ndarray
        Symmetric positive-definite matrix.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess for the solution.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.
    lambda_ : float, optional
        Regularization parameter (Tikhonov regularization).
    preconditioner : function, optional
        Function that applies the preconditioner (e.g. Incomplete Cholesky or Diagonal).
        The function must take a vector as input and return the preconditioned vector.

    Returns:
    x : ndarray
        Solution vector.

    Note:
    The Conjugate Gradient method is an iterative method that solves the linear system Ax = b. If A is not a square matrix
    we request the user to fall back on the normal equation (X^T X + lambda I) x = X^T y, where A = X^T X and b = X^T y,
    which is then solvable using the CG method.
    """
    assert A.shape[0] == A.shape[1], 'Matrix A must be square, please use the normal equation (X^T X) beta = X^T y, with A = X^T X and b = X^T y'
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if max_iter is None:
        max_iter = n

    if lambda_ > 0:
        A = A + lambda_ * np.eye(n) # Tikhonov regularization

    # Preconditioner
    if preconditioner is not None:
        M_inv = preconditioner(A)
    else:
        M_inv = lambda x: x

    x = x0
    r = b - A @ x
    z = M_inv(r)
    p = z
    rs_old = np.dot(r, z)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(r)
        rs_new = np.dot(r, z)

        if np.sqrt(rs_new) < tol:
            if verbose: print(f'Converged in {i+1} iterations')
            return x

        p = z + (rs_new / rs_old) * p
        rs_old = rs_new

    if verbose: print(f'Did not converge; reached max iterations ({max_iter})')

    return x

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
