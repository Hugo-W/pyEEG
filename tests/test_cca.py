# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:37:56 2019

@author: octave

Some test cases for CCA.
"""

import numpy as np
from pyeeg.cca import cca_nt, cca_svd
import compare_cca

# ------ A simple case for sanity checking ------
n = 100000
px = 7
py = 8
x = np.random.randn(n,px)
y = np.random.randn(n,py)

# different correlation values expected
y[:,0:2] = x[:,0:2]
y[:,2:4] = x[:,2:4] + np.random.randn(n,2)
y[:,4:6] = x[:,4:6] + 3*np.random.randn(n,2)

x = x - np.mean(x,0)
y = y - np.mean(y,0)
#-----------------------------------------------

def test_cca_healthycase():
    """
    Test the CCA implementation on a healthy case.
    """
    # CCA by SVD
    Ax, Ay, R = cca_svd(x, y)
    # CCA NT style
    A1, A2, A, B, Rnt, eigvals_x, eigvals_y = cca_nt(x, y,[1,1], None)

    # signs of coefficients can be flipped, attempting to use the same sign
    # to facilitate comparisons
    px = Ax.shape[1]
    sx = np.corrcoef(Ax, A, rowvar=False)
    sx = np.sign(np.diag(sx[:px][:, px:]))
    px = sx.shape[0]
    Ax[:, :px] = Ax[:, :px] * sx
    Ay[:, :px] = Ay[:, :px] * sx

    assert np.all(np.isclose(R, Rnt)), 'CCA NT and SVD do not give the same result!'
    xp = x @ Ax
    yp = y @ Ay
    Cx_1 = xp.T @ xp
    Cy_1 = yp.T @ yp
    Cxy_1 = yp.T @ xp

    _assert_identity_matrices(Cx_1, Cy_1, px)
    _assert_diagonal_matrices(Cxy_1, R)
    xp = x @ A
    yp = y @ B
    Cxy_2 = yp.T @ xp
    _assert_consistency(Cxy_1, Cxy_2)
    
    
def _assert_identity_matrices(Cx, Cy, px):
    assert np.all(np.isclose(Cx, np.eye(px))), 'CCA Cx is not identity!'
    assert np.all(np.isclose(Cy, np.eye(px))), 'CCA Cy is not identity!'


def _assert_diagonal_matrices(Cxy, R):
    assert np.all(np.isclose(Cxy, np.diag(np.diag(Cxy)))), 'CCA Cxy is not diagonal!'
    assert np.all(np.isclose(np.diag(Cxy), R)), 'CCA Cxy diagonal is not equal to score!'


def _assert_consistency(Cxy_1, Cxy_2):
    assert np.allclose(Cxy_1, Cxy_2), 'CCA NT/SVD is not the same!'


# Below actual examples to be plotted
# and compared, not really tests but sanity checks
if __name__ == '__main__':
    # CCA by SVD
    Ax, Ay, R = cca_svd(x, y)
    # CCA NT style
    A1, A2, A, B, Rnt, eigvals_x, eigvals_y = cca_nt(x, y,[1,1], None)
    # see comments in the function for the rationale of the plots
    compare_cca.plot(x, y, A, B, Rnt, Ax, Ay, R, ['NT style', 'SVD style'])

    # ------ A pathological / degenerate case now ------
    # x and y of dim 6 but only rank 4, and only 2 non-0 CC
    n = 100000

    # rank 4
    x = np.random.randn(n,4)
    # mixing the first 2 dimensions
    M = np.array([[1,2],[-3,5]])
    y = x[:,0:2] @ M
    # and adding a little bit of noise on one dimension
    y[:,1:] += np.random.randn(n,1)

    # adding 2 independent dimensions
    y = np.concatenate((y,10*np.random.randn(n,2)),axis=1)
    # and mixing all that in dimension 6
    # projection 4 -> 6
    Mx = np.random.randn(4,6)
    My = np.random.randn(4,6)

    x = x @ Mx
    y = y @ My

    x = x - np.mean(x,0)
    y = y - np.mean(y,0)

    Ax, Ay, R = cca_svd(x, y)
    # this example can breaks cca_nt if no regularisation is used:
    # A1, A2, A, B, Rnt, eigvals_x, eigvals_y = cca.cca_nt(x, y,[1,1], None)
    A1, A2, A, B, Rnt, eigvals_x, eigvals_y = cca_nt(x, y,[.99,.99], None)
    compare_cca.plot(x, y, A, B, Rnt, Ax, Ay, R, ['NT style', 'SVD style'])
#
#