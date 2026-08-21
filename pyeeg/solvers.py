import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
from functools import reduce
from tqdm import tqdm
import logging
from typing import Union, List, Optional

LOGGER = logging.getLogger(__name__)


def create_laplacian_matrix(n_lags: int, alpha: float = 1.0) -> np.ndarray:
    """
    Create a Laplacian matrix for smoothness constraints in quadratic regularization.
    
    The Laplacian matrix approximates the second derivative, promoting smoothness
    in the TRF coefficients across time lags.
    
    Parameters:
    ----------
    n_lags : int
        Number of time lags (dimension of the TRF).
    alpha : float, optional
        Scaling factor for the Laplacian. Default is 1.0.
        
    Returns:
    -------
    L : ndarray (n_lags, n_lags)
        Laplacian matrix for smoothness regularization.
        
    Notes:
    -----
    The Laplacian matrix L has the form:
        L[i, i-1] = -1
        L[i, i] = 2
        L[i, i+1] = -1
    for interior points, with appropriate boundary conditions.
    """
    L = np.diag(-np.ones(n_lags-1), k=1)
    L += np.diag(-np.ones(n_lags-1), k=-1)
    L += np.diag(2*np.ones(n_lags), k=0)
    # Boundary
    L[0,0] = 1
    L[-1, -1] = 1
    return alpha * L


def create_quadratic_regularizer(reg_type: str, n_lags: int, alpha: float = 1.0) -> np.ndarray:
    """
    Factory function to create quadratic regularization matrices.
    
    Parameters:
    ----------
    reg_type : str
        Type of regularization: 'smoothness' or 'laplacian'.
    n_lags : int
        Number of time lags.
    alpha : float, optional
        Regularization strength. Default is 1.0.
        
    Returns:
    -------
    M : ndarray
        Quadratic regularization matrix.
        
    Raises:
    ------
    ValueError
        If reg_type is not recognized.
    """
    if reg_type in ('smoothness', 'laplacian'):
        return create_laplacian_matrix(n_lags, alpha)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}. Use 'smoothness' or 'laplacian'.")


def svd_solver(A, b, lambda_=0., M=None, truncated_svd=False, verbose=False):
    """
    Solve the linear system Ax = b using the SVD method.
    
    This method assumes that we are solving the normal equation:
    (X^T X + lambda I + M) x = X^T y
    Thus, A = X^T X and b = X^T y.

    Parameters:
    A : ndarray
        Matrix A. Typically of shape (n_features * n_lags, n_features * n_lags) in the context of TRF.
    b : ndarray
        Right-hand side vector. Typically of shape (n_features * n_lags, n_outputs) in the context of TRF.
    lambda_ : float, optional
        Regularization parameter (Tikhonov/L2 regularization).
    M : ndarray, optional
        Quadratic regularization matrix. If provided, solves (A + M) x = b instead of (A + lambda I) x = b.
        Useful for smoothness constraints (e.g., Laplacian matrix).
    truncated_svd : bool, optional
        Whether to use the truncated SVD method. If True, lambda_ must be between 0 and 1; 
        it represents the fraction of the total variance to keep.

    Returns:
    x : ndarray
        Solution vector.
    """
    # Check symmetricity of A
    assert np.allclose(A, A.T), 'Matrix A must be symmetric'
    
    if M is not None:
        # Quadratic regularization: solve (A + M) x = b
        # Use eigendecomposition for (A + M)
        C = A + M
        eigvals, eigvecs = np.linalg.eigh(C)
        s_inv = np.diag(1 / eigvals)
        return eigvecs @ s_inv @ eigvecs.T @ b
    
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
                 M: np.ndarray = None,
                 verbose: bool = False) -> np.ndarray:
    """
    Linear regression using svd.

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
        If array, will compute betas for every regularisation parameters at once.
        Used for Tikhonov/L2 regularization.
    M : ndarray, optional
        Quadratic regularization matrix (e.g. smoothness / Laplacian). If provided,
        it REPLACES the L2 (alpha) regularization: the solution becomes
        betas = (XᵀX + M)⁻¹ Xᵀy. ``alpha`` no longer enters the solve (it only
        controls the size of the last output axis for API compatibility).
    verbose : bool, optional
        Whether to print progress information.

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
        
        # Add quadratic regularization matrix M if provided
        if M is not None:
            XtX = XtX + M
        
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
        
        eigvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        if M is not None:
            # M replaces the L2 (alpha) regularization: alpha does not enter the solve
            eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat((1. / s)[:, None], np.size(alpha), axis=1)
        else:
            eigvals_scaled[range(len(V)), range(len(V)), :] = 1 / \
                (np.repeat(s[:, None], np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        Vsreg = np.dot(V.T, eigvals_scaled) # np.diag(1/(s + alpha))
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, U.T @ XtY) #Vsreg @ Ut
    else:
        [U, s, V] = np.linalg.svd(x, full_matrices=False)
        if M is not None:
            # M replaces the L2 (alpha) regularization: solve (XᵀX + M) β = Xᵀy.
            # For a single matrix we SVD the (n_feats × n_feats) normal matrix, so
            # the projection must use Xᵀy (not y, which is (n_samples, ...)) --
            # otherwise the dimensions mismatch unless n_samples == n_feats.
            if np.ndim(x) == 2:
                XtX = x.T @ x + M
                [U, s, V] = np.linalg.svd(XtX, full_matrices=False)
            if np.ndim(y) == 3:
                XtY = np.zeros((x.shape[1], y.shape[2]), dtype=y.dtype)
                for Y in y:
                    XtY += x.T @ Y
            else:
                XtY = x.T @ y
            Uty = U.T @ XtY
        elif np.ndim(y) == 3:
            Uty = np.zeros((U.shape[1], y.shape[2]), dtype=y.dtype)
            for Y in y:
                Uty += U.T @ Y
            Uty /= len(y)
        else:
            Uty = U.T @ y

        # Broadcast all alphas (regularization param) in a 3D matrix,
        # each slice being a diagonal matrix of s/(s**2+lambda) (L2 path) or
        # 1/s (M path, where alpha does not enter the solve)
        eigvals_scaled = np.zeros((*V.shape, np.size(alpha)))
        if M is not None:
            eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat((1. / s)[:, None], np.size(alpha), axis=1)
        else:
            eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(s[:, None], np.size(alpha), axis=1) / \
                (np.repeat(s[:, None]**2, np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
        # A dot product instead of matmul allows to repeat multiplication alike across third dimension (alphas)
        Vsreg = np.dot(V.T, eigvals_scaled) # np.diag(s/(s**2 + alpha))
        # Using einsum to control which access get multiplied, again leaving alpha's dimension "untouched"
        betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty) #Vsreg @ Uty
    
    return betas


def _as_regression_segments(x, y):
    """Normalize array or segmented regression inputs for robust solvers."""
    if isinstance(x, list):
        x_segments = [np.asarray(xx, dtype=float) for xx in x]
        if isinstance(y, list):
            y_segments = [np.asarray(yy, dtype=float) for yy in y]
        else:
            y_array = np.asarray(y, dtype=float)
            if y_array.ndim != 3 or len(x_segments) != y_array.shape[0]:
                raise ValueError("Segmented X and y inputs must have matching lists.")
            y_segments = [yy for yy in y_array]
    else:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 3:
            x_segments = [x_array] * y_array.shape[0]
            y_segments = [yy for yy in y_array]
        else:
            x_segments = [x_array]
            y_segments = [y_array]

    if not x_segments or len(x_segments) != len(y_segments):
        raise ValueError("X and y must contain at least one matching segment.")

    normalized_y = []
    for xx, yy in zip(x_segments, y_segments):
        if xx.ndim != 2 or yy.ndim not in (1, 2):
            raise ValueError("Each X must be 2-D and each y must be 1-D or 2-D.")
        yy = yy[:, None] if yy.ndim == 1 else yy
        if len(xx) != len(yy):
            raise ValueError("Each X and y segment must have the same number of rows.")
        normalized_y.append(yy)

    n_chans = normalized_y[0].shape[1]
    if any(yy.shape[1] != n_chans for yy in normalized_y):
        raise ValueError("All y segments must have the same number of channels.")
    n_features = x_segments[0].shape[1]
    if any(xx.shape[1] != n_features for xx in x_segments):
        raise ValueError("All X segments must have the same number of columns.")
    return x_segments, normalized_y


def _robust_scale(residuals):
    """Estimate a positive Cauchy scale from residuals using MAD."""
    residuals = np.asarray(residuals, dtype=float)
    centered = residuals - np.median(residuals, axis=0, keepdims=True)
    scale = 1.4826 * np.median(np.abs(centered), axis=0)
    fallback = np.std(residuals, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    return np.maximum(scale, np.finfo(float).eps)


def _solve_weighted_normal_equations(x_segments, y_segments, weights, beta,
                                     alpha=0., M=None, inner_solver='svd',
                                     tol=1e-8, max_iter=None):
    """Solve one weighted least-squares subproblem for all output channels."""
    n_features = x_segments[0].shape[1]
    n_chans = y_segments[0].shape[1]
    betas = np.empty((n_features, n_chans), dtype=float)

    for channel in range(n_chans):
        xtx = np.zeros((n_features, n_features), dtype=float)
        xty = np.zeros(n_features, dtype=float)
        for xx, yy, ww in zip(x_segments, y_segments, weights):
            ww_channel = ww[:, channel] if ww.ndim == 2 else ww
            weighted_x = xx * ww_channel[:, None]
            xtx += xx.T @ weighted_x
            xty += xx.T @ (ww_channel * yy[:, channel])

        if M is not None:
            system = xtx + M
            ridge = 0.
        else:
            system = xtx
            ridge = alpha

        if inner_solver == 'cg':
            # CG operates on the assembled normal equations. The weighted
            # design is still never materialized, which keeps this path useful
            # for segmented inputs.
            betas[:, channel] = conjugate_gradient(
                system, xty, x0=beta[:, channel], tol=tol,
                max_iter=max_iter, lambda_=ridge)
        else:
            if ridge:
                system = system + ridge * np.eye(n_features)
            betas[:, channel] = np.linalg.lstsq(system, xty, rcond=None)[0]
    return betas


def _robust_irls_regress(x, y, alpha=0., M=None, loss='cauchy',
                          scale=None, max_iter=20, tol=1e-6, damping=1.0,
                          inner_solver='svd', inner_tol=1e-8,
                          inner_max_iter=None, verbose=False):
    """Fit a robust linear model using Cauchy-loss IRLS.

    The Cauchy loss is ``log(1 + (residual / scale)**2)``. Each IRLS step
    solves a weighted least-squares problem with weights
    ``1 / (1 + (residual / scale)**2)``. Array, list-of-arrays, and 3-D
    multi-segment targets are accepted.

    Returns
    -------
    betas : ndarray, shape (n_features, n_channels)
    info : dict
        Convergence metadata including ``n_iter``, ``converged`` and ``scale``.
    """
    if loss != 'cauchy':
        raise ValueError("Only loss='cauchy' is supported by robust IRLS.")
    if inner_solver not in ('svd', 'cg'):
        raise ValueError("inner_solver must be 'svd' or 'cg'.")
    if not np.isscalar(alpha):
        raise ValueError("Robust fitting requires a scalar alpha.")
    if max_iter < 1 or tol <= 0 or not 0 < damping <= 1:
        raise ValueError("max_iter must be positive, tol > 0, and damping in (0, 1].")

    x_segments, y_segments = _as_regression_segments(x, y)
    n_features = x_segments[0].shape[1]
    n_chans = y_segments[0].shape[1]

    # Use the existing regression path for a stable initial estimate.
    initial = _svd_regress(x_segments, y_segments, float(alpha), M=M,
                           verbose=False)[..., 0]
    betas = np.asarray(initial, dtype=float).reshape(n_features, n_chans)

    residual_segments = [yy - xx @ betas for xx, yy in zip(x_segments, y_segments)]
    scales = (_robust_scale(np.vstack(residual_segments)) if scale is None
              else np.full(n_chans, float(scale)))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scale must be positive and finite.")

    reg_matrix = M if M is not None else float(alpha) * np.eye(n_features)

    def objective(current):
        value = 0.
        for xx, yy in zip(x_segments, y_segments):
            residual = yy - xx @ current
            value += 0.5 * np.sum(scales[None, :] ** 2 *
                                   np.log1p((residual / scales[None, :]) ** 2))
        if reg_matrix is not None:
            value += float(np.sum(current * (reg_matrix @ current))) / 2.
        return value

    previous_value = objective(betas)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        residual_segments = [yy - xx @ betas
                             for xx, yy in zip(x_segments, y_segments)]
        weights = [1. / (1. + (residual / scales[None, :]) ** 2)
                   for residual in residual_segments]
        candidate = _solve_weighted_normal_equations(
            x_segments, y_segments, weights, betas, alpha=float(alpha), M=M,
            inner_solver=inner_solver, tol=inner_tol,
            max_iter=inner_max_iter)

        # Damping and a small backtracking safeguard help with the non-convex
        # Cauchy objective, especially when the initial OLS fit is poor.
        step = damping
        updated = (1. - step) * betas + step * candidate
        updated_value = objective(updated)
        while updated_value > previous_value and step > 1e-3:
            step *= 0.5
            updated = (1. - step) * betas + step * candidate
            updated_value = objective(updated)

        delta = np.max(np.abs(updated - betas))
        betas = updated
        if verbose:
            LOGGER.info("Robust IRLS iteration %d: delta=%g, objective=%g",
                        n_iter, delta, updated_value)
        if delta <= tol * max(1., np.max(np.abs(betas))):
            converged = True
            previous_value = updated_value
            break
        previous_value = updated_value

    return betas, {
        'n_iter': n_iter,
        'converged': converged,
        'scale': scales,
        'objective': previous_value,
    }


def _robust_least_squares_regress(x, y, scale=None, max_nfev=200,
                                  ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                  verbose=False):
    """Fit Cauchy-loss regression with SciPy's nonlinear least-squares solver.

    This reference path intentionally handles only unregularized regression.
    It is useful for small dense problems and for validating the IRLS path.
    """
    from scipy.optimize import least_squares

    x_segments, y_segments = _as_regression_segments(x, y)
    x_all = np.vstack(x_segments)
    y_all = np.vstack(y_segments)
    n_chans = y_all.shape[1]
    betas = np.empty((x_all.shape[1], n_chans), dtype=float)
    scales = (_robust_scale(y_all - x_all @ np.linalg.lstsq(x_all, y_all,
                                                             rcond=None)[0])
              if scale is None else np.full(n_chans, float(scale)))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scale must be positive and finite.")

    results = []
    for channel in range(n_chans):
        target = y_all[:, channel]
        beta0 = np.linalg.lstsq(x_all, target, rcond=None)[0]
        result = least_squares(
            lambda beta, target=target: x_all @ beta - target,
            beta0,
            loss='cauchy',
            f_scale=scales[channel],
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=2 if verbose else 0,
        )
        betas[:, channel] = result.x
        results.append(result)

    return betas, {
        'n_iter': max(result.nfev for result in results),
        'converged': all(result.success for result in results),
        'scale': scales,
        'results': results,
    }
