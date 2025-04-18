from pyeeg.solvers import svd_solver, conjugate_gradient, _lstsq_regress, _svd_regress
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
    svd_truncated_solution = svd_solver(A, b, lambda_=1-1e-8, truncated_svd=True, verbose=False)
    assert np.allclose(pseudo_inverse_solution, svd_truncated_solution), "Pseudo inverse and SVD truncated solutions do not match."
    assert np.allclose(cg_solution, svd_truncated_solution), "Conjugate gradient and SVD truncated solutions do not match."

def get_data(return_cov=False):
    """Generate data for TRF estimation tests.
    Returns a tuple of (XX, XY, ker) where XX is the lagged input matrix, XY is the output matrix, and ker is the kernel.
    """
    from pyeeg.simulate import simulate_pulse_inputs, simulate_trf_output, dummy_trf_kernel
    from pyeeg.utils import lag_matrix
    tker, ker = dummy_trf_kernel()
    t, x = simulate_pulse_inputs()
    y = simulate_trf_output(tker, ker, x)
    srate = 100
    lags = np.round(tker * srate).astype(int)
    X = lag_matrix(x, lags, filling=0., drop_missing=False)
    if return_cov:
        return X.T@X, X.T@y, ker
    return X, y, ker, tker

def test_basic_trf():
    from pyeeg.models import TRFEstimator
    X, y, ker, tker = get_data()
    # Unregularised solutions
    # This tets for the lst_sqr solver, default when alpha is None
    trf_sol =TRFEstimator(times=tker, srate=100, fit_intercept=False).fit(X[:, ::-1], y[:, None], lagged=True, drop=False).coef_.squeeze()
    # svdregress_sol = _svd_regress(X, y[:, None], alpha=0.)

    assert np.allclose(trf_sol, ker), "TRF solution do not match kernel ground truth."
    # assert np.allclose(svdregress_sol, ker), "TRF solution do not match kernel ground truth."


def test_all_solvers():
    XX, XY, ker = get_data(return_cov=True)
    # Unregularised solutions
    pseudo_inverse_solution = np.linalg.pinv(XX) @ XY
    cg_solution = conjugate_gradient(XX, XY, np.zeros_like(XY), lambda_=0.0)
    svd_solution = svd_solver(XX, XY, lambda_=0., truncated_svd=False, verbose=False)
    lstsq_sol = _lstsq_regress(XX, XY)

    assert np.allclose(pseudo_inverse_solution, ker), "Pinv solution do not match kernel ground truth."
    assert np.allclose(cg_solution, ker),             "CG solution do not match kernel ground truth."
    assert np.allclose(svd_solution, ker),            "SVD solution do not match kernel ground truth."
    assert np.allclose(lstsq_sol, ker),               "Lstsq solution do not match kernel ground truth."

