# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:06:07 2019

@author: octave
"""

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, Ax1, Ay1, R1, Ax2, Ay2, R2, impl_names):

    # signs of coefficients can be flipped, attempting to use the same sign
    # to facilitate comparisons
    px = Ax1.shape[1]

    sx = np.corrcoef(Ax1, Ax2, rowvar=False)
    sx = np.sign(np.diag(sx[0:px,px:]))
    px = sx.shape[0]
    Ax1[:,0:px] = Ax1[:,0:px] * sx
    Ay1[:,0:px] = Ay1[:,0:px] * sx

    fig = plt.figure()

    # common colourbar scale
    m = np.max([np.abs(Ax1).max(), np.abs(Ay1).max(),
                np.abs(Ax2).max(), np.abs(Ay2).max()]);

    nCol = 6

    for iRow, cx, cy, r, plt_title in zip([1, 2], [Ax1, Ax2], [Ay1, Ay2], [R1, R2], impl_names):

        # projections should be orthonormal and cross-projection orthogonal
        xp = x @ cx
        yp = y @ cy

        # expecting diagonal identity matrices Cx & Cy and diagonal Cxy with scores on
        # the diagonal
        Cx = xp.T @ xp
        Cy = yp.T @ yp
        Cxy = yp.T @ xp

        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 1)
        ax.imshow(Cx, vmin=0, vmax=1)
        if iRow == 1:
            ax.set_title(r'$A_x x^T x A_x$')
        ax.set_ylabel(plt_title)
    #    fig.colorbar(im, ax=ax)


        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 2)
        ax.imshow(Cy, vmin=0, vmax=1)
        if iRow == 1:
            ax.set_title(r'$A_y y^T y A_y$')
    #    fig.colorbar(im, ax=ax)


        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 3)
        plt.imshow(Cxy, vmin=0, vmax=1)
        if iRow == 1:
            ax.set_title(r'$A_y y^T x A_x$')


        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 4)
        ax.plot(r)
        ax.plot(np.diag(Cxy))
        ax.set_ylim((np.min([r.min(),-0.05]),np.max([r.max(),1.05])))
        asp = np.diff(ax.get_xlim()) / np.diff(ax.get_ylim())
        ax.set_aspect(asp[0])

        if iRow == 1:
            ax.set_title('Correlations')

        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 5)
        ax.imshow(cx, vmin=-m, vmax=m)
        if iRow == 1:
            ax.set_title(r'$A_x$')

        ax = fig.add_subplot(2, nCol, nCol*(iRow-1) + 6)
        ax.imshow(cy, vmin=-m, vmax=m)
        if iRow == 1:
            ax.set_title(r'$A_y$')

    fig.tight_layout()
    plt.show()
    return fig