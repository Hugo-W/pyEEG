from pyeeg.solvers import svd_solver, conjugate_gradient
import numpy as np

def test_compare_solvers():
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    A = np.random.rand(5, 5)
    # A bit of multicolinearity in A, not rank deficient though, but towards ill-conditioning
    A[0] = A[2] * 0.1 + np.random.rand(5)
    A = A @ A.T
    b = np.random.rand(5)
    x0 = np.zeros_like(b)
    # Regularised solutions
    cg_solution = conjugate_gradient(A, b, x0, lambda_=.00001)
    svd_solution = svd_solver(A, b, lambda_=0.00001)
    assert np.allclose(cg_solution, svd_solution), "Conjugate gradient and SVD solutions do not match."
    # Unregularised solutions
    pseudo_inverse_solution = np.linalg.pinv(A) @ b
    cg_solution = conjugate_gradient(A, b, x0, lambda_=0.0)
    svd_truncated_solution = svd_solver(A, b, lambda_=1-1e-8, truncated_svd=True, verbose=True)
    assert np.allclose(pseudo_inverse_solution, svd_truncated_solution), "Pseudo inverse and SVD truncated solutions do not match."
    assert np.allclose(cg_solution, svd_truncated_solution), "Conjugate gradient and SVD truncated solutions do not match."