# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:09:24 2019

@author: Karen
"""

import logging
import mne
import numpy as np
import sys
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from mne.decoding import BaseEstimator
from sklearn.cross_decomposition import CCA
from pyeeg.utils import lag_matrix, lag_span, lag_sparse, is_pos_def, find_knee_point
from pyeeg.vizu import topoplot_array
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize as zscore

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def cca_nt(x, y, threshs, knee_point):
    # A, B: transform matrices
    # R: r scores
    # can normalise the data
    m = x.shape[1]
    # Build covariance matrix: C=[x,y]'*[x,y]
    
    if isinstance(y, list):
        n = y[0].shape[1]
        C = np.zeros((m + n,m + n))
        # create list of X for all y's
        all_x = [x for i in range(len(y))]
        x_cov = sum(list(map(lambda a,b: a.T @ b, all_x, all_x)))        
        y_cov = sum(list(map(lambda a,b: a.T @ b, y, y)))
        xy_cov = sum(list(map(lambda a,b: a.T @ b, all_x, y)))
        C[:m,:m] = x_cov
        C[m:,m:] = y_cov
        C[:m,m:] = xy_cov
        C[m:,:m] = xy_cov.T      
    else:
        C = np.concatenate([x, y], axis=1).T @ np.concatenate([x, y], axis=1)
    
    # PCA on X.T X to get sphering matrix A1 and on Y.T*Y for A2
    As = []
    Eigvals = []
    for idx, C_temp in enumerate([C[:m, :m], C[m:, m:]]):
        Val, Vec = np.linalg.eigh(C_temp)               # get eigval & eigvec
        if not is_pos_def(C_temp):
            discard = np.argwhere(Val < 0)
            if not len(discard) == 0:
                Val = Val[max(discard)[0]+1:]
                Vec = Vec[:,max(discard)[0]+1:]
        Val, Vec = Val[::-1], Vec[:, ::-1]
        if knee_point is not None:
            find_knee_point()
            print('knee_point used')    
        keep = np.cumsum(Val)/sum(Val) <= threshs[idx]   # only keep components over certain percentage of variance
        topcs = Vec[:, keep]                        # corresponding vecs
        Val = Val[keep]
        exp = 1-1e-12
        Val = Val**exp
        As.append(topcs @ np.diag(np.sqrt(1/Val)))
        Eigvals.append(Val)
    A1, A2 = As
    eigvals_x, eigvals_y = Eigvals 
    
    # create new C = Amix.T*C*Amix
    AA = np.zeros((A1.shape[0] + A2.shape[0], A1.shape[1] + A2.shape[1]))
    AA[:A1.shape[0], :A1.shape[1]] = A1
    AA[A1.shape[0]:, A1.shape[1]:] = A2
    C = AA.T @ C @ AA
    
    N = np.min((np.size(A1,1), np.size(A2,1)))    # number of canonical components
    
    # PCA on Cnew
    Val, Vec = np.linalg.eigh(C)
    Val, Vec = Val[::-1], Vec[:, ::-1]
    
    A = A1 @ Vec[:np.size(A1,1),:N]*np.sqrt(2)      # keeping only N first PCs
    B = A2 @ Vec[np.size(A1,1):,:N]*np.sqrt(2)
    R = Val[:N] - 1
    
    return A1, A2, A, B, R, eigvals_x, eigvals_y

class CCA_Estimator(BaseEstimator):
    
    """Canonocal Correlation (CCA) Estimator Class.

    Attributes
    ----------
    lags : 1d-array
        Array of `int`, corresponding to lag in samples at which the TRF coefficients are computed
    times : 1d-array
        Array of `float`, corresponding to lag in seconds at which the TRF coefficients are computed
    srate : float
        Sampling rate
    fit_intercept : bool
        Whether a column of ones should be added to the design matrix to fit an intercept
    intercept_ : 1d array (nchans, )
        Intercepts
    coef_ : ndarray (nlags, nfeats, nchans)
        Actual TRF coefficients
    n_feats_ : int
        Number of word level features in TRF
    n_chans_: int
        Number of EEG channels in TRF
    feat_names_ : list
        Names of each word level features
    Notes
    -----
    Attributes with a `_` suffix are only set once the TRF has been fitted on EEG data

   """
        
    def __init__(self, times =(0.,), tmin=None, tmax=None, srate=1., fit_intercept=True):
        
        if tmin and tmax:
            LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(-tmax, -tmin, srate=srate)
            self.times = -self.lags[::-1] / srate
        else:
            self.lags = lag_sparse(times, srate)
            self.times = np.asarray(times)
            
        self.srate = srate
        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coefStim_ = None
        self.coefResponse_ = None
        self.score_ = None
        self.eigvals_x = None
        self.eigvals_y = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.sklearn_TRF_ = None
        self.tempX_path_ = None
        self.tempy_path_ = None
        
    
    def fit(self, X, y, cca_implementation='nt', thresh_x=None, normalise=True, thresh_y=None, n_comp=2, knee_point=None, drop=True, feat_names=()):
        """ Fit CCA model.
        
        X : ndarray (nsamples x nfeats)
            Array of features (time-lagged)
        y : ndarray (nsamples x nchans)
            EEG data

        Returns
        -------
        coef_ : ndarray (nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """
        if isinstance(y, list):
            print('y is a list. CCA will be implemented more efficiently.')
            self.n_chans_ = y[0].shape[1]
        else:
            self.n_chans_ = y.shape[1]
        self.n_feats_ = X.shape[1]
        if feat_names:
            self.feat_names_ = feat_names

        # Creating lag-matrix droping NaN values if necessary
        if drop:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=True)

            # Droping rows of NaN values in y
            if any(np.asarray(self.lags) < 0):
                drop_top = abs(min(self.lags))
                y = y[drop_top:, :]
            if any(np.asarray(self.lags) > 0):
                drop_bottom = abs(max(self.lags))
                y = y[:-drop_bottom, :]
        else:
            X = lag_matrix(X, lag_samples=self.lags, filling=0.)
        
        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])          
        if thresh_x is None:
            if thresh_y is None:
                thresh_x = 0.999
                thresh_y = 0.999
            else:
                thresh_x = thresh_y
        if thresh_y is None:
            thresh_y = thresh_x
        threshs = np.asarray([thresh_x, thresh_y])
        
        if cca_implementation == 'nt':
            A1, A2, A, B, R, eigvals_x, eigvals_y = cca_nt(X, y, threshs, knee_point)
            # Reshaping and getting coefficients
            if self.fit_intercept:
                self.intercept_ = A[0, :]
                A = A[1:, :]

            self.coefResponse_ = B
            self.score_ = R
            self.eigvals_x = eigvals_x
            self.eigvals_y =eigvals_y

        if cca_implementation == 'sklearn':
            cca_skl = CCA(n_components=n_comp)
            cca_skl.fit(X, y)
            A = cca_skl.x_rotations_
            if self.fit_intercept:
                self.intercept_ = A[0, :]
                A = A[1:, :]

            self.coefResponse_ = cca_skl.y_rotations_
            score = np.diag(np.corrcoef(cca_skl.x_scores_, cca_skl.y_scores_, rowvar=False)[:n_comp, n_comp:])
            self.score_ = score
            self.sklearn_TRF_ = cca_skl.coef_

        # save the matrix X and y to save memory
        if self.fit_intercept:
            X = X[:, 1:]
        if sys.platform.startswith("win"):
            tmpdir = os.environ["TEMP"]
        else:
            tmpdir = os.environ["TMPDIR"]
        np.save(os.path.join(tmpdir,'temp_X'), X)
        if isinstance(y, list):
            np.save(os.path.join(tmpdir,'temp_y'), np.asarray(y).reshape((len(y)*y[0].shape[0], y[0].shape[1])))
        else:
            np.save(os.path.join(tmpdir,'temp_y'), y)
        self.tempX_path_ = os.path.join(tmpdir,'temp_X')
        self.tempy_path_ = os.path.join(tmpdir,'temp_y')
        
        self.coefStim_ = np.reshape(A, (len(self.lags), self.n_feats_, self.coefResponse_.shape[1]))
            
        
    def plot_time_filter(self, n_comp=1, dim=[0]):
        """Plot the TRF of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        if n_comp < 6:
            for c in range(n_comp):
                for d in range(len(dim)):
                    plt.plot(self.times, self.coefStim_[:,dim[d],c],label='CC #%s, dim: %s' % ((c+1), dim[d]))
        else:
            for c in range(5):
                for d in range(len(dim)):
                    plt.plot(self.times, self.coefStim_[:,dim[d],c],label='CC #%s, dim: %s' % ((c+1), dim[d]))
            for c in range(5,n_comp):
                for d in range(len(dim)):
                    plt.plot(self.times, self.coefStim_[:,dim[d],c])
        if self.feat_names_:
            plt.title('Time filter for {:s}'.format(self.feat_names_[0]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim([-max(np.abs(self.coefStim_[:,dim,:n_comp].flatten())), max(np.abs(self.coefStim_[:,dim,:n_comp].flatten()))]);
            
    def plot_spatial_filter(self, pos, n_comp=1):
        """Plot the topo of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(self.coefResponse_, pos, n_topos=n_comp, titles=titles)
        mne.viz.tight_layout()
        
        
    def plot_corr(self, pos, n_comp=1):
        """Plot the correlation between the EEG component waveform and the EEG channel waveform.
        Parameters
        ----------
        """
        X = np.load(self.tempX_path_+'.npy')
        all_x = np.concatenate([X]*5)
        y = np.load(self.tempy_path_+'.npy')
        coefStim_ = self.coefStim_.reshape((self.coefStim_.shape[0] * self.coefStim_.shape[1], self.coefStim_.shape[2]))

        r = np.zeros((64,n_comp))
        for c in range(n_comp):
            eeg_proj = y @ self.coefResponse_[:, c]
            env_proj = all_x @ coefStim_[:, c]
            for i in range(64):
                r[i,c] = np.corrcoef(y[:,i], eeg_proj)[0,1]
            cc_corr = np.corrcoef(eeg_proj, env_proj)[0,1]
        
        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(r, pos, n_topos=n_comp, titles=titles)
        mne.viz.tight_layout()
        
    def plot_activation_map(self, pos, n_comp=1):
        """Plot the activation map from the spatial filter.
        Parameters
        ----------
        """   
        y = np.load(self.tempy_path_+'.npy')
        
        if n_comp <= 0:
            print('Invalid number of components, must be a positive integer.')
        s_hat = y @ self.coefResponse_
        sigma_eeg = y.T @ y
        sigma_reconstr = s_hat.T @ s_hat
        a_map = sigma_eeg @ self.coefResponse_ @ np.linalg.inv(sigma_reconstr)
        
        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(a_map, pos, n_topos=n_comp, titles=titles)
        mne.viz.tight_layout()
        
    def plot_compact_time(self, n_comp=2, dim=0):
        plt.imshow(self.coefStim_[:, dim, :n_comp].T, aspect='auto', origin='bottom', extent=[self.times[0], self.times[-1], 0, n_comp])
        plt.colorbar()
        plt.ylabel('Components')
        plt.xlabel('Time (ms)')
        plt.title('Dimension #{:d}'.format(dim+1))

    def plot_all_dim_time(self, n_comp=0, n_dim=2):
        n_comp = range(n_comp)
        n_rows = len(n_comp) // 2 + len(n_comp)%2
        fig = plt.figure(figsize=(10, 20), constrained_layout=False)
        outer_grid = fig.add_gridspec(n_rows, 2, wspace=0.1, hspace=0.1)
        bottoms, tops, _, _ = outer_grid.get_grid_positions(fig)
        for c, coefs in enumerate(self.coefStim_.swapaxes(0,2)[n_comp]):
            inner_grid = outer_grid[c].subgridspec(1, 1)
            ax = plt.Subplot(fig, inner_grid[0])
            vmin = np.min(coefs)
            vmax = np.max(coefs)
            im = ax.imshow(coefs, aspect=0.04, origin='bottom',
                           extent=[self.times[0], self.times[-1], 0, n_dim],
                           vmin=vmin, vmax=vmax)
            if c // 2 != n_rows-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (s)')
            if c%2!=0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('Dimension')
            ax.set(title=('CC #{:d}'.format(c+1)))
            fig.add_subplot(ax)
        # Colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.5 - 0.05, 0.01, 0.1])
        fig.colorbar(im, cax=cbar_ax, shrink=0.6)