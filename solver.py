import numpy as np
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix

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

# Example usage
if __name__ == "__main__":
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    A = np.random.rand(5, 5)
    # A bit of multilinerity in A, slightly rank deficient:
    A[0] = A[2] * 0.1 + np.random.rand(5)
    A = A @ A.T
    b = np.random.rand(5)
    x0 = np.zeros_like(b)
    # Compare with pseudo-inverse solution
    cg_solution = conjugate_gradient(A, b, x0, lambda_=.00001)
    pseudo_inverse_solution = np.linalg.pinv(A) @ b
    svd_solution = svd_solver(A, b, lambda_=0.00001)
    svd_truncated_solution = svd_solver(A, b, lambda_=1-1e-8, truncated_svd=True, verbose=True)
    print("CG Solution:            \t", cg_solution)
    print("Pseudo-inverse Solution:\t", pseudo_inverse_solution)
    print("SVD Solution:           \t", svd_solution)
    print("SVD Truncated Solution: \t", svd_truncated_solution)
    # They should be equal

    
