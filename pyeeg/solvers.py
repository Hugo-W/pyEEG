"""Regression solvers for TRF (temporal response function) fitting.

This module provides the building blocks used to fit linear regression
models on (possibly segmented / multi-epoch) data:

- ``svd_solver`` and :class:`SVDSolver`: ridge / truncated-SVD regression
  via the SVD of the normal matrix ``XᵀX`` (or of the tall design matrix
  ``X`` with ``use_full_svd=True``).
- :class:`LSTSQSolver`: plain least-squares regression via
  ``numpy.linalg.lstsq`` on the accumulated normal equations.
- ``conjugate_gradient``, ``block_conjugate_gradient`` and
  :class:`ConjugateGradientSolver`: iterative solves of the (regularized)
  normal equations, optionally preconditioned.
- :class:`IRLSSolver`: robust regression using Cauchy-loss iteratively
  reweighted least squares.
- :class:`ScipyRobustSolver`: reference robust Cauchy-loss regression
  built on ``scipy.optimize.least_squares`` (unregularized, small dense
  problems; validates the IRLS path).
- Regularizers: ``create_laplacian_matrix`` and
  ``create_quadratic_regularizer`` build quadratic (smoothness) penalty
  matrices; ``incomplete_cholesky_preconditioner`` and
  ``diagonal_preconditioner`` build preconditioners for CG.
- The :class:`Solver` abstract base class and :class:`SolverResult`
  dataclass define the common interface: every solver accepts
  ``(X, y, alpha, M)`` and returns a :class:`SolverResult`.
"""
import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
from functools import reduce
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, List, Optional

from ._logging import LOGGER


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
            LOGGER.info(f'Keeping {n_components} components (out of {len(s)})')
            LOGGER.info(f'Variance explained: {s[:n_components].sum() / s.sum()}')
            LOGGER.info(f"Singular values: {s[:n_components]}")
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
            if verbose: LOGGER.info(f'Converged in {i+1} iterations')
            return x

        p = z + (rs_new / rs_old) * p
        rs_old = rs_new

    if verbose: LOGGER.info(f'Did not converge; reached max iterations ({max_iter})')

    return x


def block_conjugate_gradient(A, B, X0=None, tol=1e-10, max_iter=None,
                             lambda_=0., verbose=False):
    """Block Conjugate Gradient: solve A X = B for multiple right-hand sides.

    Solves all channels simultaneously using Frobenius inner products,
    eliminating the per-channel Python loop.  Converges in a single set
    of iterations (governed by the hardest channel), but amortizes the
    matrix-matrix products A @ P across all channels.

    Parameters
    ----------
    A : ndarray (n, n)
        Symmetric positive-definite matrix.
    B : ndarray (n, k)
        Right-hand sides (k channels).
    X0 : ndarray (n, k), optional
        Initial guess. Defaults to zeros.
    tol : float
        Convergence tolerance on the global residual norm.
    max_iter : int, optional
        Maximum iterations. Defaults to n.
    lambda_ : float
        Tikhonov regularization (added to A).

    Returns
    -------
    X : ndarray (n, k)
        Solution for all channels.
    """
    n, k = B.shape
    if X0 is None:
        X0 = np.zeros((n, k))
    if max_iter is None:
        max_iter = n

    if lambda_ > 0:
        A = A + lambda_ * np.eye(n)

    X = X0
    R = B - A @ X
    P = R.copy()
    RR_old = np.einsum('ij,ij->', R, R)  # Frobenius inner product <R, R>

    for i in range(max_iter):
        AP = A @ P
        PP = np.einsum('ij,ij->', P, AP)
        if PP < np.finfo(float).eps * np.finfo(float).eps:
            break
        alpha = RR_old / PP
        X = X + alpha * P
        R = R - alpha * AP
        RR_new = np.einsum('ij,ij->', R, R)

        if np.sqrt(RR_new) < tol:
            if verbose:
                LOGGER.info(f'Block CG converged in {i+1} iterations')
            return X

        beta = RR_new / RR_old
        P = R + beta * P
        RR_old = RR_new

    if verbose:
        LOGGER.info(f'Block CG did not converge; reached max iterations ({max_iter})')
    return X


def _as_regression_segments(x, y):
    """Normalize array or segmented regression inputs for robust solvers.

    Converts ``x``/``y`` into matching lists of 2-D segments: a 2-D ``x``
    with a 2-D ``y`` becomes a single-element list; a 2-D ``x`` with a 3-D
    ``y`` (n_epochs, n_samples, n_channels) is repeated for each epoch; a
    list ``x`` is paired element-wise with a list ``y`` or the epochs of a
    3-D ``y``.

    Parameters
    ----------
    x : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels), 3-D array
        (n_epochs, n_samples, n_channels), or list of 1-D/2-D arrays (one
        per segment).

    Returns
    -------
    x_segments : list of ndarray
        Each element is a 2-D design matrix (n_samples, n_features).
    y_segments : list of ndarray
        Each element is a 2-D target (n_samples, n_channels).

    Raises
    ------
    ValueError
        If the inputs are inconsistent (mismatched list lengths, wrong
        dimensionality, unequal sample counts, or differing numbers of
        channels / features across segments).
    """
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
    """Estimate a positive Cauchy scale from residuals using MAD.

    Uses the median absolute deviation (MAD) scaled by 1.4826 (the
    consistency factor for normally distributed data). Falls back to the
    standard deviation where the MAD degenerates to zero, and clamps the
    result to a positive machine epsilon.

    Parameters
    ----------
    residuals : ndarray
        Residuals of shape (n_samples,) or (n_samples, n_channels).

    Returns
    -------
    scale : float or ndarray
        Robust scale estimate. A float when ``residuals`` is 1-D, otherwise
        an array of shape (n_channels,).
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
        squeeze = True
    else:
        squeeze = False
    centered = residuals - np.median(residuals, axis=0, keepdims=True)
    scale = 1.4826 * np.median(np.abs(centered), axis=0)
    fallback = np.std(residuals, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, fallback)
    scale = np.maximum(scale, np.finfo(float).eps)
    return scale[0] if squeeze else scale


def _solve_weighted_normal_equations(x_segments, y_segments, weights, beta,
                                     alpha=0., M=None, inner_solver='svd',
                                     tol=1e-8, max_iter=None):
    """Solve one weighted least-squares subproblem for all output channels.

    Assembles per-channel weighted normal equations and solves them in a
    single batched call to ``np.linalg.solve`` (or CG per channel when
    ``inner_solver='cg'``), avoiding the Python-level per-channel loop
    overhead of separate factorizations.
    """
    n_features = x_segments[0].shape[1]
    # Handle 1-D y (single channel, from per-channel IRLS)
    if y_segments[0].ndim == 1:
        y_segments = [yy[:, None] for yy in y_segments]
        squeeze = True
    else:
        squeeze = False
    n_chans = y_segments[0].shape[1]

    # Stack per-channel systems: (n_chans, n_features, n_features) and
    # (n_chans, n_features).  This replaces the per-channel Python loop
    # with a single batched solve.
    systems = np.empty((n_chans, n_features, n_features), dtype=float)
    rhs = np.empty((n_chans, n_features), dtype=float)

    for channel in range(n_chans):
        xtx = np.zeros((n_features, n_features), dtype=float)
        xty = np.zeros(n_features, dtype=float)
        for xx, yy, ww in zip(x_segments, y_segments, weights):
            ww_channel = ww[:, channel] if ww.ndim == 2 else ww
            weighted_x = xx * ww_channel[:, None]
            xtx += xx.T @ weighted_x
            xty += xx.T @ (ww_channel * yy[:, channel])

        if M is not None:
            systems[channel] = xtx + M
        elif alpha:
            systems[channel] = xtx + alpha * np.eye(n_features)
        else:
            systems[channel] = xtx
        rhs[channel] = xty

    if inner_solver == 'cg':
        # CG is a vector algorithm — still need per-channel calls, but
        # the system assembly above is already batched.
        betas = np.empty((n_features, n_chans), dtype=float)
        ridge = 0. if M is not None else alpha
        for channel in range(n_chans):
            betas[:, channel] = conjugate_gradient(
                systems[channel], rhs[channel], x0=beta[:, channel], tol=tol,
                max_iter=max_iter, lambda_=0.)  # ridge already in system
    else:
        # Single batched solve across all channels
        betas = np.linalg.solve(systems, rhs[..., None])[..., 0].T
    return betas


@dataclass
class SolverResult:
    """Result container for solver runs.

    Parameters
    ----------
    betas : ndarray
        Estimated coefficients. Shape is (n_features, n_channels) for most
        solvers, or (n_features, n_channels, n_alphas) when ``alpha`` is
        array-like (e.g. :class:`SVDSolver`).
    info : dict or None, optional
        Optional solver metadata (e.g. ``n_iter``, ``converged``, ``scale``
        for the robust solvers). Default is None.
    """
    betas: np.ndarray
    info: Optional[dict] = None


class Solver(ABC):
    """Abstract base class for regression solvers.

    All solvers accept (X, y, alpha, M) and return a SolverResult.
    X can be a 2-D array or a list of 2-D arrays (segments).
    y can be a 2-D array, 3-D array (multi-epoch), or list of 2-D arrays.

    Parameters
    ----------
    X : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels) or (n_samples, n_channels, n_epochs) or list.
    alpha : float or array-like
        Regularization strength(s). When M is provided, alpha is ignored in
        the solve (it only controls output axis size for API compatibility).
    M : ndarray or None
        Quadratic regularization matrix. If provided, REPLACES L2 (alpha)
        regularization: the solution becomes betas = (X^T X + M)^{-1} X^T y.

    Returns
    -------
    result : SolverResult
        Contains `betas` (ndarray) and `info` (dict or None).
    """
    @abstractmethod
    def solve(self, X, y, alpha=0.0, M=None):
        """Solve the regression problem and return the coefficients.

        Parameters
        ----------
        X : ndarray or list of ndarray
            Design matrix (n_samples, n_features) or list of segments.
        y : ndarray or list of ndarray
            Target (n_samples, n_channels), (n_samples, n_channels, n_epochs),
            or list of arrays (one per segment).
        alpha : float or array-like, optional
            Regularization strength(s). When ``M`` is provided, ``alpha`` is
            ignored in the solve (it only controls the output axis size for
            API compatibility). Default is 0.0.
        M : ndarray or None, optional
            Quadratic regularization matrix. If provided, replaces L2
            (``alpha``) regularization. Default is None.

        Returns
        -------
        result : SolverResult
            Contains ``betas`` (ndarray of estimated coefficients) and
            ``info`` (dict or None).
        """
        pass


class SVDSolver(Solver):
    """Linear regression using the singular value decomposition (SVD).

    Solves the regularized normal equations ``(XᵀX + lambda I + M) beta =
    Xᵀy`` via SVD. By default the SVD is computed on the small normal matrix
    ``XᵀX`` (n_features, n_features), which is much faster for tall matrices
    (n_samples >> n_features) and gives identical results to factorizing the
    tall design matrix ``X`` directly.

    Parameters
    ----------
    verbose : bool, optional
        Whether to print progress information. Default is False.
    truncated : bool, optional
        If True, ``alpha`` is reinterpreted as the fraction of total variance
        to keep (between 0 and 1). Instead of Tikhonov regularization (which
        shrinks all components), truncated SVD keeps the top-k components
        explaining ``alpha`` of the variance and discards the rest entirely.
        Incompatible with ``M`` (quadratic regularizer). Default is False.
    use_full_svd : bool, optional
        If True, SVD the tall design matrix ``X`` (n_samples, n_features)
        for higher numerical precision. If False (default), SVD the small
        normal matrix ``XᵀX`` (n_features, n_features) which is much faster
        for tall matrices (n_samples >> n_features) and gives identical
        results. Default is False.

    Notes
    -----
    A warning is shown in the case where n_features > n_samples; if so the
    user should rather use partial regression.
    """
    def __init__(self, verbose=False, truncated=False, use_full_svd=False):
        self.verbose = verbose
        self.truncated = truncated
        self.use_full_svd = use_full_svd

    def solve(self, X, y, alpha=0.0, M=None):
        """Solve the SVD regression and return the coefficients.

        ``X`` may be a 2-D array or a list of 2-D arrays (segments /
        trials). When a list is given (with possibly different n_samples),
        the covariance matrices are accumulated across trials; the number of
        samples must then be the same in ``X`` and ``y`` per trial. If ``y``
        is a 3-D array (n_epochs, n_samples, n_chans), the normal equations
        are accumulated across epochs.

        Parameters
        ----------
        X : ndarray (n_samples, n_features) or list of such
            Design matrix, or list of segments to accumulate.
        y : ndarray (n_samples, n_channels) or list of such
            Target. If ``y`` is a list of arrays, each element is treated as
            an individual subject / segment and the ``betas`` coefficients
            are computed on the accumulated covariance matrices.
        alpha : float or array-like, optional
            Regularization parameter (Tikhonov/L2). If array-like, betas are
            computed for every regularization value at once. When ``M`` is
            provided, ``alpha`` no longer enters the solve (it only controls
            the size of the last output axis for API compatibility). Default
            is 0.0.
        M : ndarray, optional
            Quadratic regularization matrix (e.g. smoothness / Laplacian).
            If provided, it REPLACES the L2 (``alpha``) regularization: the
            solution becomes ``betas = (XᵀX + M)⁻¹ Xᵀy``. Default is None.

        Returns
        -------
        result : SolverResult
            ``betas`` has shape (n_features, n_channels) or
            (n_features, n_channels, len(alpha)) when ``alpha`` is
            array-like; ``info`` is None.
        """
        # cast alpha in ndarray
        if np.isscalar(alpha):
            alpha = np.asarray([alpha], dtype=float)
        else:
            alpha = np.asarray(alpha)

        if self.truncated:
            if M is not None:
                raise ValueError(
                    "Truncated SVD is incompatible with quadratic regularizer M."
                )
            if not np.all((alpha > 0) & (alpha <= 1)):
                raise ValueError(
                    "For truncated SVD, alpha must be in (0, 1] "
                    "(fraction of variance to keep)."
                )
        else:
            if not isinstance(X, list) and np.ndim(X) == 2:
                if X.shape[0] < X.shape[1]:
                    LOGGER.warning("Less samples than features! The linear problem is not stable in that form. Consider using partial regression instead.")
            try:
                assert np.all(alpha >= 0), "Alpha must be positive"
            except AssertionError:
                raise ValueError

        if (len(X) == len(y)) and np.ndim(X[0])==2: # will accumulate covariances
            assert all([xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(X, y)]), "Inconsistent trial lengths!"
            XtX = reduce(lambda x, y: x + y, [xx.T @ xx for xx in X])

            # Add quadratic regularization matrix M if provided
            if M is not None:
                XtX = XtX + M

            [U, s, V] = np.linalg.svd(XtX, full_matrices=False) # here V = U.T
            XtY = np.zeros((XtX.shape[0], y[0].shape[1]), dtype=y[0].dtype)
            segments = zip(X, y)
            if self.verbose:
                segments = tqdm(segments, total=len(X), leave=False, desc='Covariance accumulation')
            for xx, yy in segments:
                XtY += xx.T @ yy
            # XtY /= len(x) # NO: IT SHOULD BE A SUM

            #betas = U @ np.diag(1/(s + alpha)) @ U.T @ XtY

            eigvals_scaled = np.zeros((*V.shape, np.size(alpha)))
            if self.truncated:
                # Truncated SVD: keep top components explaining alpha fraction
                # of total variance.  SVD here is of XtX, so variance = s.
                cum_var = np.cumsum(s) / np.sum(s)
                for a_idx, a_val in enumerate(alpha):
                    k = np.searchsorted(cum_var, a_val) + 1
                    k = min(k, len(s))
                    if self.verbose:
                        LOGGER.info(
                            "Truncated SVD: keeping %d/%d components "
                            "(alpha=%.3f, variance=%.4f)",
                            k, len(s), a_val, cum_var[k - 1],
                        )
                    eigvals_scaled[:k, :k, a_idx] = np.diag(1.0 / s[:k])
            elif M is not None:
                # M replaces the L2 (alpha) regularization: alpha does not enter the solve
                eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat((1. / s)[:, None], np.size(alpha), axis=1)
            else:
                eigvals_scaled[range(len(V)), range(len(V)), :] = 1 / \
                    (np.repeat(s[:, None], np.size(alpha), axis=1) + np.repeat(alpha[:, None].T, len(s), axis=0))
            Vsreg = np.dot(V.T, eigvals_scaled) # np.diag(1/(s + alpha))
            betas = np.einsum('...jk, jl -> ...lk', Vsreg, U.T @ XtY) #Vsreg @ Ut
        else:
            # Single matrix path.
            # By default, SVD the small XtX (n_features × n_features) instead
            # of the tall X (n_samples × n_features) — 10–15× faster for
            # typical MEG/EEG data (n_samples >> n_features) with identical
            # results.  Set use_full_svd=True for the original tall-matrix SVD
            # (slightly higher precision, but rarely needed).
            if not self.use_full_svd:
                # --- Fast path: SVD of XtX (same as segment accumulation) ---
                XtX = X.T @ X
                if M is not None:
                    XtX = XtX + M
                [U, s, V] = np.linalg.svd(XtX, full_matrices=False)

                # Compute XtY
                if np.ndim(y) == 3:
                    n_chans = y.shape[2]
                    XtY = np.zeros((X.shape[1], n_chans), dtype=y.dtype)
                    for Y in y:
                        XtY += X.T @ Y
                else:
                    XtY = X.T @ y

                Uty = U.T @ XtY

                eigvals_scaled = np.zeros((*V.shape, np.size(alpha)))
                if self.truncated:
                    cum_var = np.cumsum(s) / np.sum(s)
                    for a_idx, a_val in enumerate(alpha):
                        k = np.searchsorted(cum_var, a_val) + 1
                        k = min(k, len(s))
                        if self.verbose:
                            LOGGER.info(
                                "Truncated SVD: keeping %d/%d components "
                                "(alpha=%.3f, variance=%.4f)",
                                k, len(s), a_val, cum_var[k - 1],
                            )
                        eigvals_scaled[:k, :k, a_idx] = np.diag(1.0 / s[:k])
                elif M is not None:
                    eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(
                        (1. / s)[:, None], np.size(alpha), axis=1)
                else:
                    eigvals_scaled[range(len(V)), range(len(V)), :] = 1 / \
                        (np.repeat(s[:, None], np.size(alpha), axis=1) +
                         np.repeat(alpha[:, None].T, len(s), axis=0))
                Vsreg = np.dot(V.T, eigvals_scaled)
                betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty)

            else:
                # --- Full SVD path: SVD of X (tall matrix, higher precision) ---
                [U, s, V] = np.linalg.svd(X, full_matrices=False)
                if M is not None:
                    if np.ndim(X) == 2:
                        XtX = X.T @ X + M
                        [U, s, V] = np.linalg.svd(XtX, full_matrices=False)
                    if np.ndim(y) == 3:
                        XtY = np.zeros((X.shape[1], y.shape[2]), dtype=y.dtype)
                        for Y in y:
                            XtY += X.T @ Y
                    else:
                        XtY = X.T @ y
                    Uty = U.T @ XtY
                elif np.ndim(y) == 3:
                    Uty = np.zeros((U.shape[1], y.shape[2]), dtype=y.dtype)
                    for Y in y:
                        Uty += U.T @ Y
                    Uty /= len(y)
                else:
                    Uty = U.T @ y

                eigvals_scaled = np.zeros((*V.shape, np.size(alpha)))
                if self.truncated:
                    cum_var = np.cumsum(s ** 2) / np.sum(s ** 2)
                    for a_idx, a_val in enumerate(alpha):
                        k = np.searchsorted(cum_var, a_val) + 1
                        k = min(k, len(s))
                        if self.verbose:
                            LOGGER.info(
                                "Truncated SVD: keeping %d/%d components "
                                "(alpha=%.3f, variance=%.4f)",
                                k, len(s), a_val, cum_var[k - 1],
                            )
                        eigvals_scaled[:k, :k, a_idx] = np.diag(1.0 / s[:k])
                elif M is not None:
                    eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(
                        (1. / s)[:, None], np.size(alpha), axis=1)
                else:
                    eigvals_scaled[range(len(V)), range(len(V)), :] = np.repeat(
                        s[:, None], np.size(alpha), axis=1) / \
                        (np.repeat(s[:, None]**2, np.size(alpha), axis=1) +
                         np.repeat(alpha[:, None].T, len(s), axis=0))
                Vsreg = np.dot(V.T, eigvals_scaled)
                betas = np.einsum('...jk, jl -> ...lk', Vsreg, Uty)

        return SolverResult(betas, None)


class LSTSQSolver(Solver):
    """Linear regression using least squares (``numpy.linalg.lstsq``).

    Accumulates the normal equations ``XᵀX`` and ``Xᵀy`` (across segments /
    epochs when needed) and solves them with ``numpy.linalg.lstsq``. No
    regularization is applied; ``alpha`` and ``M`` are accepted for API
    compatibility with :class:`Solver` but do not modify the solution.

    Notes
    -----
    A warning is shown in the case where n_features > n_samples; if so the
    user should rather use partial regression.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def solve(self, X, y, alpha=0.0, M=None):
        """Solve the least-squares regression and return the coefficients.

        ``X`` may be a 2-D array or a list of 2-D arrays (segments /
        trials). When a list is given, the covariance matrices are
        accumulated across trials; the number of samples must then be the
        same in ``X`` and ``y`` per trial. ``alpha`` and ``M`` are accepted
        for API compatibility but do not modify the solution.

        Parameters
        ----------
        X : ndarray (n_samples, n_features) or list of such
            Design matrix, or list of segments to accumulate.
        y : ndarray (n_samples, n_channels) or list of such
            Target. If ``y`` is a list of arrays, each element is treated as
            an individual subject / segment and the ``betas`` coefficients
            are computed on the accumulated covariance matrices.
        alpha : float or array-like, optional
            Accepted for API compatibility; not used by this solver.
            Default is 0.0.
        M : ndarray, optional
            Accepted for API compatibility; not used by this solver.
            Default is None.

        Returns
        -------
        result : SolverResult
            ``betas`` has shape (n_features, n_channels); ``info`` is None.
        """
        if not isinstance(X, list) and np.ndim(X) == 2:
            if X.shape[0] < X.shape[1]:
                LOGGER.warning("Less samples than features! The linear problem is not stable in that form. Consider using partial regression instead.")

        if (len(X) == len(y)) and np.ndim(X[0])==2: # will accumulate covariances
            assert all([xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(X, y)]), "Inconsistent trial lengths!"
            XtX = reduce(lambda x, y: x + y, [xx.T @ xx for xx in X])
            XtY = np.zeros((XtX.shape[0], y[0].shape[1]), dtype=y[0].dtype)
            segments = zip(X, y)
            if self.verbose:
                segments = tqdm(segments, total=len(X), leave=False, desc='Covariance accumulation')
            for xx, yy in segments:
                XtY += xx.T @ yy

            betas = np.linalg.lstsq(XtX, XtY)[0]
        elif np.ndim(y) == 3:
            # 3-D y: (n_epochs, n_samples, n_chans) — accumulate normal equations
            # across epochs. Accumulating BOTH XtX and XtY cancels the n_epochs
            # factor, so the solution matches _svd_regress (averaged) semantics.
            n_features = X.shape[1]
            XtX = np.zeros((n_features, n_features), dtype=float)
            n_chans = y.shape[2]
            XtY = np.zeros((n_features, n_chans), dtype=float)
            for yy in y:
                XtX += X.T @ X
                XtY += X.T @ yy
            betas = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
            return SolverResult(betas, None)
        else:
            betas = np.linalg.lstsq(X, y)[0]
        return SolverResult(betas, None)


class ConjugateGradientSolver(Solver):
    """Regression solver using Conjugate Gradient on normal equations.

    Computes ``XᵀX`` and ``Xᵀy``, then solves
    ``(XᵀX + alpha*I + M) beta = Xᵀy`` using the Conjugate Gradient method.

    Parameters
    ----------
    tol : float, optional
        Convergence tolerance for the conjugate gradient. Default is 1e-10.
    max_iter : int or None, optional
        Maximum number of iterations. If None, defaults to the number of
        features. Default is None.
    preconditioner : callable or None, optional
        Function that builds a preconditioner from the system matrix
        ``A`` (e.g. :func:`incomplete_cholesky_preconditioner` or
        :func:`diagonal_preconditioner`). If None, no preconditioning is
        applied. Default is None.
    verbose : bool, optional
        Whether to log progress information. Default is False.
    """
    def __init__(self, tol=1e-10, max_iter=None, preconditioner=None, verbose=False):
        self.tol = tol
        self.max_iter = max_iter
        self.preconditioner = preconditioner
        self.verbose = verbose

    def solve(self, X, y, alpha=0.0, M=None):
        """Solve the regression problem with block conjugate gradient.

        Accumulates the normal equations ``XᵀX`` and ``Xᵀy`` across
        segments / epochs when needed, optionally adds the quadratic
        regularizer ``M``, and solves for all output channels
        simultaneously with :func:`block_conjugate_gradient`.

        Parameters
        ----------
        X : ndarray or list of ndarray
            Design matrix (n_samples, n_features) or list of segments
            (with possibly different n_samples).
        y : ndarray or list of ndarray
            Target (n_samples, n_channels), 3-D array
            (n_epochs, n_samples, n_channels), or list of arrays (one per
            segment).
        alpha : float, optional
            Tikhonov/L2 regularization strength. When ``M`` is provided,
            ``alpha`` is ignored in the solve. Default is 0.0.
        M : ndarray or None, optional
            Quadratic regularization matrix added to ``XᵀX``. If provided,
            it replaces the L2 (``alpha``) regularization. Default is None.

        Returns
        -------
        result : SolverResult
            ``betas`` has shape (n_features, n_channels); ``info`` is None.
        """
        # Handle list of segments
        if isinstance(X, list) and len(X) == len(y) and np.ndim(X[0]) == 2:
            assert all(xtr.shape[0] == ytr.shape[0] for xtr, ytr in zip(X, y)), "Inconsistent trial lengths!"
            XtX = reduce(lambda a, b: a + b, [xx.T @ xx for xx in X])
            n_chans = y[0].shape[1] if y[0].ndim == 2 else 1
            n_features = XtX.shape[0]
            XtY = np.zeros((n_features, n_chans), dtype=float)
            for xx, yy in zip(X, y):
                yy = yy[:, None] if yy.ndim == 1 else yy
                if yy.shape[1] != n_chans:
                    raise ValueError("All y segments must have the same number of channels.")
                XtY += xx.T @ yy
        else:
            X = np.asarray(X)
            y = np.asarray(y)
            if y.ndim == 3:
                # 3-D y: (n_epochs, n_samples, n_chans) — accumulate normal
                # equations across epochs (both XtX and XtY), matching the
                # semantics of _svd_regress (which averages) and LSTSQSolver.
                n_features = X.shape[1]
                n_chans = y.shape[2]
                XtX = np.zeros((n_features, n_features), dtype=float)
                XtY = np.zeros((n_features, n_chans), dtype=float)
                for yy in y:
                    XtX += X.T @ X
                    XtY += X.T @ yy
            else:
                XtX = X.T @ X
                XtY = X.T @ y
                n_features = XtX.shape[0]
                n_chans = XtY.shape[1] if XtY.ndim == 2 else 1

        if M is not None:
            XtX = XtX + M

        if XtY.ndim == 1:
            XtY = XtY[:, None]
            n_chans = 1

        # Block CG: solve all channels simultaneously using Frobenius
        # inner products, eliminating the per-channel Python loop.
        betas = block_conjugate_gradient(
            XtX, XtY, tol=self.tol, max_iter=self.max_iter,
            lambda_=float(alpha) if alpha else 0.0,
            verbose=self.verbose
        )

        return SolverResult(betas, None)


def _irls_single_channel(x_segments, y_channel, initial_beta, alpha, M,
                         scale, max_iter, tol, damping, inner_solver,
                         inner_tol, inner_max_iter, verbose=False):
    """Run the full IRLS loop for a single output channel.

    Each channel is independent: residuals, weights, scale, convergence
    are all per-channel.  This function is designed to be called in
    parallel across channels.
    """
    n_features = x_segments[0].shape[1]
    beta = np.asarray(initial_beta, dtype=float).copy()

    # Per-channel residuals and scale
    residual_segments = [yy - xx @ beta for xx, yy in zip(x_segments, y_channel)]
    s = float(scale) if scale is not None else float(_robust_scale(
        np.concatenate(residual_segments)))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("scale must be positive and finite.")

    reg_matrix = M if M is not None else float(alpha) * np.eye(n_features)

    def objective(current):
        value = 0.
        for xx, yy in zip(x_segments, y_channel):
            residual = yy - xx @ current
            value += 0.5 * s ** 2 * np.sum(np.log1p((residual / s) ** 2))
        if reg_matrix is not None:
            value += float(np.sum(current * (reg_matrix @ current))) / 2.
        return value

    previous_value = objective(beta)
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        residual_segments = [yy - xx @ beta for xx, yy in zip(x_segments, y_channel)]
        weights = [1. / (1. + (residual / s) ** 2) for residual in residual_segments]

        candidate = _solve_weighted_normal_equations(
            x_segments, y_channel, weights, beta[:, None], alpha=float(alpha),
            M=M, inner_solver=inner_solver, tol=inner_tol,
            max_iter=inner_max_iter)[:, 0]

        # Damping and backtracking
        step = damping
        updated = (1. - step) * beta + step * candidate
        updated_value = objective(updated)
        while updated_value > previous_value and step > 1e-3:
            step *= 0.5
            updated = (1. - step) * beta + step * candidate
            updated_value = objective(updated)

        delta = np.max(np.abs(updated - beta))
        beta = updated
        if verbose:
            LOGGER.info("IRLS ch: iter %d delta=%g obj=%g", n_iter, delta, updated_value)
        if delta <= tol * max(1., np.max(np.abs(beta))):
            converged = True
            previous_value = updated_value
            break
        previous_value = updated_value

    return beta, {
        'n_iter': n_iter,
        'converged': converged,
        'scale': s,
        'objective': previous_value,
    }


class IRLSSolver(Solver):
    """Fit a robust linear model using Cauchy-loss IRLS.

    The Cauchy loss is ``log(1 + (residual / scale)**2)``. Each IRLS step
    solves a weighted least-squares problem with weights
    ``1 / (1 + (residual / scale)**2)``. Array, list-of-arrays, and 3-D
    multi-segment targets are accepted.

    Each channel is solved independently with its own convergence criterion
    (some channels may converge faster than others).  When ``n_jobs > 1``,
    channels are processed in parallel using joblib.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs for per-channel IRLS.  Default 1 (sequential).
        Use -1 for all available cores.  When parallel, each worker uses a
        single BLAS thread to avoid oversubscription.
    loss : str
        Only 'cauchy' is supported.
    scale : float or None
        Fixed scale for the Cauchy loss.  If None, estimated from residuals.
    max_iter : int
        Maximum IRLS iterations per channel.
    tol : float
        Convergence tolerance (per-channel).
    damping : float
        Damping factor for the IRLS step (0, 1].
    inner_solver : str
        Inner solver for weighted normal equations: 'svd' or 'cg'.
    inner_tol : float
        Tolerance for the inner CG solver.
    inner_max_iter : int or None
        Max iterations for the inner CG solver.
    verbose : bool
        Print per-iteration progress.

    Returns
    -------
    result : SolverResult
        betas : ndarray (n_features, n_channels)
        info : dict with n_iter, converged, scale, objective
    """
    def __init__(self, loss='cauchy', scale=None, max_iter=20, tol=1e-6,
                 damping=1.0, inner_solver='svd', inner_tol=1e-8,
                 inner_max_iter=None, n_jobs=1, verbose=False):
        self.loss = loss
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.inner_solver = inner_solver
        self.inner_tol = inner_tol
        self.inner_max_iter = inner_max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

    def solve(self, X, y, alpha=0.0, M=None):
        """Fit the robust Cauchy-loss IRLS model and return coefficients.

        Accepts array, list-of-arrays, and 3-D multi-segment targets (see
        :func:`_as_regression_segments`). Each channel is solved
        independently with its own convergence criterion; channels may be
        processed in parallel when ``n_jobs > 1``.

        Parameters
        ----------
        X : ndarray or list of ndarray
            Design matrix (n_samples, n_features) or list of segments.
        y : ndarray or list of ndarray
            Target (n_samples, n_channels), 3-D array
            (n_epochs, n_samples, n_channels), or list of arrays (one per
            segment).
        alpha : float, optional
            Tikhonov/L2 regularization strength. Must be scalar. When ``M``
            is provided, ``alpha`` is ignored in the solve. Default is 0.0.
        M : ndarray or None, optional
            Quadratic regularization matrix. If provided, it replaces the L2
            (``alpha``) regularization: the solution becomes
            ``betas = (XᵀX + M)⁻¹ Xᵀy``. Default is None.

        Returns
        -------
        result : SolverResult
            ``betas`` has shape (n_features, n_channels). ``info`` is a dict
            with keys ``n_iter`` (max across channels), ``converged`` (True
            if all channels converged), ``scale`` (array of per-channel
            scales), and ``objective`` (sum of final objectives).
        """
        if self.loss != 'cauchy':
            raise ValueError("Only loss='cauchy' is supported by robust IRLS.")
        if self.inner_solver not in ('svd', 'cg'):
            raise ValueError("inner_solver must be 'svd' or 'cg'.")
        if not np.isscalar(alpha):
            raise ValueError("Robust fitting requires a scalar alpha.")
        if self.max_iter < 1 or self.tol <= 0 or not 0 < self.damping <= 1:
            raise ValueError("max_iter must be positive, tol > 0, and damping in (0, 1].")

        x_segments, y_segments = _as_regression_segments(X, y)
        n_features = x_segments[0].shape[1]
        n_chans = y_segments[0].shape[1]

        # Batched initial estimate for all channels at once.
        initial = SVDSolver(verbose=False).solve(
            x_segments, y_segments, float(alpha), M=M).betas[..., 0]
        betas_init = np.asarray(initial, dtype=float).reshape(n_features, n_chans)

        # Per-channel scale estimates
        residual_segments = [yy - xx @ betas_init for xx, yy in zip(x_segments, y_segments)]
        scales = (_robust_scale(np.vstack(residual_segments)) if self.scale is None
                  else np.full(n_chans, float(self.scale)))
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("scale must be positive and finite.")

        # Per-channel y slices: each is a list of 1-D arrays (one per segment)
        y_per_channel = [
            [yy[:, ch] for yy in y_segments]
            for ch in range(n_chans)
        ]

        # Run IRLS per channel (sequential or parallel)
        if self.n_jobs == 1:
            results = [
                _irls_single_channel(
                    x_segments, y_per_channel[ch], betas_init[:, ch],
                    float(alpha), M, scales[ch], self.max_iter, self.tol,
                    self.damping, self.inner_solver, self.inner_tol,
                    self.inner_max_iter, self.verbose
                )
                for ch in range(n_chans)
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor
            from threadpoolctl import threadpool_limits

            # Limit BLAS threads to 1 per worker to avoid oversubscription.
            # For the small normal matrices in IRLS, single-threaded BLAS
            # is actually faster than multi-threaded (less coordination
            # overhead), and it lets the ThreadPoolExecutor parallelise
            # across channels without competition for CPU cores.
            with threadpool_limits(limits=1), ThreadPoolExecutor(
                max_workers=self.n_jobs if self.n_jobs > 0 else None
            ) as pool:
                results = list(pool.map(
                    lambda ch: _irls_single_channel(
                        x_segments, y_per_channel[ch], betas_init[:, ch],
                        float(alpha), M, scales[ch], self.max_iter, self.tol,
                        self.damping, self.inner_solver, self.inner_tol,
                        self.inner_max_iter, self.verbose
                    ),
                    range(n_chans),
                ))

        # Aggregate results
        betas = np.column_stack([r[0] for r in results])
        infos = [r[1] for r in results]

        return SolverResult(betas, {
            'n_iter': max(info['n_iter'] for info in infos),
            'converged': all(info['converged'] for info in infos),
            'scale': np.array([info['scale'] for info in infos]),
            'objective': sum(info['objective'] for info in infos),
        })


class ScipyRobustSolver(Solver):
    """Fit Cauchy-loss regression with SciPy's nonlinear least-squares solver.

    This reference path intentionally handles only unregularized regression.
    It is useful for small dense problems and for validating the IRLS path.

    Parameters
    ----------
    scale : float or None, optional
        Fixed scale for the Cauchy loss. If None, estimated from residuals
        via :func:`_robust_scale`. Default is None.
    max_nfev : int, optional
        Maximum number of function evaluations per channel passed to
        ``scipy.optimize.least_squares``. Default is 200.
    ftol : float, optional
        Function tolerance for ``least_squares``. Default is 1e-8.
    xtol : float, optional
        Variable tolerance for ``least_squares``. Default is 1e-8.
    gtol : float, optional
        Gradient tolerance for ``least_squares``. Default is 1e-8.
    verbose : bool, optional
        Whether to print per-channel solver progress. Default is False.
    """
    def __init__(self, scale=None, max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=False):
        self.scale = scale
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.verbose = verbose

    def solve(self, X, y, alpha=0.0, M=None):
        """Fit the robust Cauchy-loss regression and return coefficients.

        Fits each output channel with ``scipy.optimize.least_squares``
        using the ``'cauchy'`` loss. Only unregularized regression is
        supported: ``alpha`` and ``M`` are accepted for API compatibility
        but ignored.

        Parameters
        ----------
        X : ndarray or list of ndarray
            Design matrix (n_samples, n_features) or list of segments.
        y : ndarray or list of ndarray
            Target (n_samples, n_channels) or list of arrays (one per
            segment).
        alpha : float or array-like, optional
            Accepted for API compatibility; not used by this solver.
            Default is 0.0.
        M : ndarray, optional
            Accepted for API compatibility; not used by this solver.
            Default is None.

        Returns
        -------
        result : SolverResult
            ``betas`` has shape (n_features, n_channels). ``info`` is a dict
            with keys ``n_iter`` (max nfev across channels), ``converged``
            (True if all channels succeeded), ``scale`` (array of per-channel
            scales), and ``results`` (list of
            ``scipy.optimize.OptimizeResult`` objects).
        """
        from scipy.optimize import least_squares

        x_segments, y_segments = _as_regression_segments(X, y)
        x_all = np.vstack(x_segments)
        y_all = np.vstack(y_segments)
        n_chans = y_all.shape[1]
        betas = np.empty((x_all.shape[1], n_chans), dtype=float)
        scales = (_robust_scale(y_all - x_all @ np.linalg.lstsq(x_all, y_all,
                                                                 rcond=None)[0])
                  if self.scale is None else np.full(n_chans, float(self.scale)))
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
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                verbose=2 if self.verbose else 0,
            )
            betas[:, channel] = result.x
            results.append(result)

        return SolverResult(betas, {
            'n_iter': max(result.nfev for result in results),
            'converged': all(result.success for result in results),
            'scale': scales,
            'results': results,
        })


def _lstsq_regress(x, y, verbose=False):
    """Linear regression using lstsq. See :class:`LSTSQSolver`.

    Parameters
    ----------
    x : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels) or list of arrays (one per segment).
    verbose : bool, optional
        Whether to print progress information. Default is False.

    Returns
    -------
    betas : ndarray (n_features, n_channels)
        Estimated coefficients.
    """
    return LSTSQSolver(verbose=verbose).solve(x, y).betas

def _svd_regress(x, y, alpha, M=None, verbose=False):
    """Linear regression using SVD. See :class:`SVDSolver`.

    Parameters
    ----------
    x : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels) or list of arrays (one per segment).
    alpha : float or array-like
        Regularization strength(s) (Tikhonov/L2).
    M : ndarray, optional
        Quadratic regularization matrix. If provided, replaces the L2
        (``alpha``) regularization. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is False.

    Returns
    -------
    betas : ndarray (n_features, n_channels) or (n_features, n_channels, len(alpha))
        Estimated coefficients.
    """
    return SVDSolver(verbose=verbose).solve(x, y, alpha, M=M).betas

def _robust_irls_regress(x, y, alpha=0., M=None, loss='cauchy',
                          scale=None, max_iter=20, tol=1e-6, damping=1.0,
                          inner_solver='svd', inner_tol=1e-8,
                          inner_max_iter=None, verbose=False):
    """Fit robust linear model using Cauchy-loss IRLS. See :class:`IRLSSolver`.

    Parameters
    ----------
    x : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels), 3-D array, or list of arrays (one
        per segment).
    alpha : float, optional
        Tikhonov/L2 regularization strength (scalar). Default is 0.
    M : ndarray, optional
        Quadratic regularization matrix. If provided, replaces the L2
        (``alpha``) regularization. Default is None.
    loss : str, optional
        Only 'cauchy' is supported. Default is 'cauchy'.
    scale : float or None, optional
        Fixed scale for the Cauchy loss. If None, estimated from residuals.
        Default is None.
    max_iter : int, optional
        Maximum IRLS iterations per channel. Default is 20.
    tol : float, optional
        Convergence tolerance (per-channel). Default is 1e-6.
    damping : float, optional
        Damping factor for the IRLS step (0, 1]. Default is 1.0.
    inner_solver : str, optional
        Inner solver for weighted normal equations: 'svd' or 'cg'.
        Default is 'svd'.
    inner_tol : float, optional
        Tolerance for the inner CG solver. Default is 1e-8.
    inner_max_iter : int or None, optional
        Max iterations for the inner CG solver. Default is None.
    verbose : bool, optional
        Whether to print per-iteration progress. Default is False.

    Returns
    -------
    betas : ndarray (n_features, n_channels)
        Estimated coefficients.
    info : dict
        Solver metadata (n_iter, converged, scale, objective).
    """
    solver = IRLSSolver(loss=loss, scale=scale, max_iter=max_iter, tol=tol,
                       damping=damping, inner_solver=inner_solver,
                       inner_tol=inner_tol, inner_max_iter=inner_max_iter,
                       verbose=verbose)
    result = solver.solve(x, y, alpha=alpha, M=M)
    return result.betas, result.info

def _robust_least_squares_regress(x, y, scale=None, max_nfev=200,
                                  ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=False):
    """Fit Cauchy-loss regression with SciPy. See :class:`ScipyRobustSolver`.

    Parameters
    ----------
    x : ndarray or list of ndarray
        Design matrix (n_samples, n_features) or list of segments.
    y : ndarray or list of ndarray
        Target (n_samples, n_channels) or list of arrays (one per segment).
    scale : float or None, optional
        Fixed scale for the Cauchy loss. If None, estimated from residuals.
        Default is None.
    max_nfev : int, optional
        Maximum number of function evaluations per channel. Default is 200.
    ftol : float, optional
        Function tolerance for ``scipy.optimize.least_squares``.
        Default is 1e-8.
    xtol : float, optional
        Variable tolerance for ``scipy.optimize.least_squares``.
        Default is 1e-8.
    gtol : float, optional
        Gradient tolerance for ``scipy.optimize.least_squares``.
        Default is 1e-8.
    verbose : bool, optional
        Whether to print per-channel solver progress. Default is False.

    Returns
    -------
    betas : ndarray (n_features, n_channels)
        Estimated coefficients.
    info : dict
        Solver metadata (n_iter, converged, scale, results).
    """
    solver = ScipyRobustSolver(scale=scale, max_nfev=max_nfev,
                               ftol=ftol, xtol=xtol, gtol=gtol, verbose=verbose)
    result = solver.solve(x, y)
    return result.betas, result.info
