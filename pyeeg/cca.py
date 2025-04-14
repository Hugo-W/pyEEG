# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:09:24 2019

@author: Karen
"""

import logging
# import mne
import numpy as np
import sys
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import CCA
from pyeeg.utils import lag_matrix, lag_span, lag_sparse, is_pos_def, find_knee_point
from pyeeg.vizu import topoplot_array, topomap
import matplotlib.pyplot as plt
from pyeeg.preprocess import create_filterbank, apply_filterbank

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
        print(m)
        print(n)
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


def cca_svd(x, y, opt={}):
    """CCA by SVD.

    Implements CCA between x and y with regularisation options as specified in
    opt.

    If y is a list of matrices, computes CCA between the concatenated
    elements of y and a repeated version of x (generic model).

    Parameters
    ----------
    x : ndarray (nsamples x ndims_x)
        Multivariate data

    y : ndarray (nsamples x ndims_y) or list of ndarray (all of the same shape)
        Multivariate data

    opt : dictionary
        Regularisation options for x and y. opt['x'] and opt['y'] are
        dictionaries containing options that will be used for their respective
        PCA. See :func:`reg_eigen` for valid options.

    Returns
    -------
    Ax, Ay : ndarrays (ndims_x x nCC and ndims_y x nCC)
        Coefficients such that (y @ Ay).T @ (x @ Ax) is diagognal with the
        largest correlation values possible on the diagonal.

    R: ndarray
        Correlation values between the projections of x and y (i.e.
        R[i] = (y.T @ Ay[i]).T @ (x @ Ax[i]) )

    Remark
    -------
    x and y are assumed to be zero-mean column-wise.

    Example
    -------
    >>> x = np.random.randn(1000,10)
    >>> y = np.random.randn(1000,8)
    >>> opt = {'x': {'nKeep': 3}, 'y': {'var': 0.99}}
    >>> Ax, Ay, R = cca_svd(x, y, opt)

    CCA between x and y, while keeping only 3 top PCs for x and dimensions
    accounting for 99 % the total variance for y.
    """

    # TODO: if y is a list, add option to switch between generic model and
    # one CCA per element of y?

    # default input
    if not 'x' in opt:
        opt['x'] = {}

    if not 'y' in opt:
        opt['y'] = {}

    # same as below with a for loop, courtesy of Hugo
    #    S, V = {}, {}
    #    for k, mat in zip(opt.keys(), [x, y]):
    #        S[k] = regEigen(mat, opt[k])

    # regularised PCA
    S, V = reg_eigen(x,opt['x'])
    O, N = reg_eigen(y,opt['y'])

    S = 1 / np.sqrt(S)
    O = 1 / np.sqrt(O)

    if isinstance(y, list):
        ytx = np.zeros((y[0].shape[1],x.shape[1]))
        for m in y:
            ytx += m.T @ x
    else:
        ytx = y.T @ x

    # this does an SVD on C given by (matrix multiplications):
    # C = diag(O) * N' * y' * x * V * diag(S)
    P, Q, R = np.linalg.svd((O[:,np.newaxis] * N.T) @ ytx @ (V * S), full_matrices=False)

    # P, Q, R = np.linalg.svd(X) such that: X = P * Q * R (and not R.T)
    # hence R needs to be transposed below
    # (singular values already sorted in descending order)
    Ax = (V * S) @ R.T
    Ay = (N * O) @ P

    return Ax, Ay, Q


def reg_eigen(x, opt={}):
    """Regularised PCA.

    Does an eigenvalue decompostion x' * x = V * S * V' and returns matrix V
    and vector S sorted by decreasing eigenvalues and with some regularisation
    as specfied in opt (reg_eigen will always discard at least degenerate
    dimensions).

    If x is a list of matrices, does the same on the summed covariance matrix
    of its elements (i.e. PCA on the concatenated elements of x ; hence: all
    elements must all have equal 2nd dimension).

    Parameters
    ----------
    x : ndarray (nsamples x ndims) or list of ndarray  (all of the same shape)
        Multivariate data

    opt : dictionary
        Regularisation options for x and y. Possible options:

            key 'cond', val: float
                Keep all eigenvalues e_i that verify:
                    (max(eigen) / eigen_i) < opt['cond']
            key 'nKeep', val: int
                Number of eigenvalues to keep
            key 'var', val : float
                Fraction of the total variance to keep
            key 'absTol', val : float
                Discard eigenvalues smaller than absTol

    Returns
    -------
    S : ndarray (nPC,)
        Sorted eigenvalues (decreasing)

    V : (ndims x nPC)
        Rotation matrix (eigenvectors) -- sorted as S

    Remark
    -------
    x and y are assumed to be zero-mean column-wise. regEigen does not apply
    any normalisation.

    Examples
    -------
    reg_eigen(x,{'cond': 1e6}) # keep eigenvals up to a condition number of 1e6
    reg_eigen(x,{'nKeep': 10}) # keep the 10 PCs with largest eigenvalues
    reg_eigen(x,{'var': 0.99}) # keep 99 % of variance
    reg_eigen(x,{'absTol': 1e-6}) # discard dimension with variance < 1e-6
    """

    # TODO: add checks on input validity?

    if isinstance(x, list):
        xtx = np.zeros((x[0].shape[1],x[0].shape[1]))
        for m in x:
            xtx += m.T @ m

        S, V = np.linalg.eigh(xtx)

    else:
        S, V = np.linalg.eigh(x.T @ x)

    # eigh sort eigenvalues in increasing order
    # reorder in decreasing order
    # NB: there may be (small) negative eigenvalues in the case of degenerate
    # matrices, we'll discard these here
    rx = np.sum(S < 0)
    S = np.flip(S[rx:])
    V = np.flip(V[:, rx:], 1)

    # always remove (at least) numerical zeros by default
    # same default tolerance as the one used to compute rank in
    # np.linalg.matrix_rank
    tol = S[0] * x.shape[1] * np.finfo(S.dtype).eps
    rx = np.sum(S > tol) # keep the rx first elements

    # regularisation:
    # keep a custom number of dimensions
    if opt:

        # condition number
        if 'cond' in opt:
            # since all eigenvals are now > 0
            rc = np.searchsorted(S[0] / S, opt['cond'])

        # explicit number of dimensions to keep
        elif 'nKeep' in opt:
            rc = opt['nKeep']

        # fraction of total variance
        elif 'var' in opt:
            rc = np.searchsorted(np.cumsum(S) / np.sum(S), opt['var'])

        # absolute tolerance value
        elif 'absTol' in opt:
            rc = np.sum(S > opt['absTol'])

        # TODO: add ridge regularisation?

        rx = np.min((rx, rc))

    # drop dimensions
    V = V[:, :rx]
    S = S[:rx]

    return S,V


class CCA_Estimator(BaseEstimator):

    """Canonical Correlation (CCA) Estimator Class.

    Attributes
    ----------
    xlags : 1d-array
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

    def __init__(self, times=(0.,), tmin=None, tmax=None, filterbank=False, freqs=(0.,), srate=1., fit_intercept=True):

        self.srate = srate
        if filterbank:
            self.f_bank = create_filterbank(freqs=freqs, srate=self.srate, N=2, rs=3)
        
        else:
            if tmin and tmax:
                LOGGER.info("Will use xlags spanning form tmin to tmax.\nTo use individual xlags, use the `times` argument...")
                self.xlags = lag_span(tmin, tmax, srate=srate)[::-1]
                self.xtimes = self.xlags[::-1] / srate
            else:
                self.xtimes = np.asarray(times)
                self.xlags = lag_sparse(self.xtimes, srate)[::-1]
            

        self.fit_intercept = fit_intercept
        self.fitted = False
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coefStim_ = None
        self.coefResponse_ = None
        self.score_ = None
        self.lag_y = False
        self.ylags = None
        self.ytimes = None
        self.eigvals_x = None
        self.eigvals_y = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.sklearn_TRF_ = None
        self.f_bank_freqs = freqs
        self.f_bank_used = filterbank
        self.tempX_path_ = None
        self.tempy_path_ = None


    def fit(self, X, y, cca_implementation='nt', thresh_x=None, normalise=True, thresh_y=None, n_comp=2, knee_point=None, drop=True, y_already_dropped=False, lag_y=False, ylags=(0.,), feat_names=(), opt_cca_svd={}):
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
            
        # Creating filterbank
        if self.f_bank_used:
            temp_X = apply_filterbank(X,self.f_bank)
            X = np.reshape(temp_X,(X.shape[0],temp_X.shape[0]*temp_X.shape[2]))
            if isinstance(y, list):
                filterbank_y = []
                for subj in range(len(y)):
                    # NEED TO CHANGE TO drop_missing=True
                    temp = apply_filterbank(y[subj],self.f_bank)
                    temp_y = np.reshape(temp,(y[subj].shape[0],temp.shape[0]*temp.shape[2]))
                    filterbank_y.append(temp_y)
            else:
                # NEED TO CHANGE TO drop_missing=True
                temp = apply_filterbank(y,self.f_bank)
                filterbank_y = np.reshape(temp,(y.shape[0],temp.shape[0]*temp.shape[2]))
            y = filterbank_y  
        else:
        # Creating lag-matrix droping NaN values if necessary
            if drop:
                X = lag_matrix(X, lag_samples=self.xlags, drop_missing=True)
                
                if not y_already_dropped:
                    # Droping rows of NaN values in y
                    if isinstance(y, list):
                        temp = []
                        for yy in y:
                            if any(np.asarray(self.xlags) < 0):
                                drop_top = abs(min(self.xlags))
                                yy = yy[drop_top:, :] if yy.ndim == 2 else yy[:, drop_top:, :]
                            if any(np.asarray(self.xlags) > 0):
                                drop_bottom = abs(max(self.xlags))
                                yy = yy[:-drop_bottom, :] if yy.ndim == 2 else yy[:, :-drop_bottom, :]
                            temp.append(yy)
                        y = temp
                    else:
                        if any(np.asarray(self.xlags) < 0):
                            drop_top = abs(min(self.xlags))
                            y = y[drop_top:, :] if y.ndim == 2 else y[:, drop_top:, :]
                        if any(np.asarray(self.xlags) > 0):
                            drop_bottom = abs(max(self.xlags))
                            y = y[:-drop_bottom, :] if y.ndim == 2 else y[:, :-drop_bottom, :]
                        
                if lag_y:
                    self.lag_y = True
                    self.ytimes = np.asarray(ylags)
                    self.ylags = -lag_sparse(self.ytimes, self.srate)[::-1]
                    if isinstance(y, list):
                        lagged_y = []
                        for subj in range(len(y)):
                            # NEED TO CHANGE TO drop_missing=True
                            temp = lag_matrix(y[subj], lag_samples=self.ylags, drop_missing=False, filling=0.)
                            lagged_y.append(temp)
                    else:
                        # NEED TO CHANGE TO drop_missing=True
                        lagged_y = lag_matrix(y , lag_samples=self.ylags, drop_missing=False, filling=0.)
                        print(lagged_y.shape)
                    y = lagged_y    
            else:
                X = lag_matrix(X, lag_samples=self.xlags, filling=0.)
                if lag_y:
                    self.lag_y = True
                    self.ytimes = np.asarray(ylags)
                    self.ylags = -lag_sparse(self.ytimes, self.srate)[::-1]
                    if isinstance(y, list):
                        lagged_y = []
                        for subj in range(len(y)):
                            temp = lag_matrix(y[subj], lag_samples=self.ylags, filling=0.)
                            lagged_y.append(temp)
                    else:
                        lagged_y = lag_matrix(y , lag_samples=self.ylags, filling=0.)
                    y = lagged_y    
        
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

        if cca_implementation == 'svd':
            Ax, Ay, R = cca_svd(X, y, opt_cca_svd)
            # Reshaping and getting coefficients
            if self.fit_intercept:
                self.intercept_ = Ax[0, :]
                A = Ax[1:, :]
            else:
                A = Ax

            self.coefResponse_ = Ay
            self.score_ = R

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
        
        if self.f_bank_used:
            self.coefStim_ = np.reshape(A, (len(self.f_bank_freqs), self.n_feats_, self.coefResponse_.shape[1]))
        else:
            self.coefStim_ = np.reshape(A, (len(self.xlags), self.n_feats_, self.coefResponse_.shape[1]))
            self.coefStim_ = self.coefStim_[::-1, :, :]

    def transform(self, transform_x=True, transform_y=False, comp=0):
        """ Transform X and Y using the coefficients
        """
        X = np.load(self.tempX_path_+'.npy')
#        y = np.load(self.tempy_path_+'.npy')
#        if len(y) > len(X):
#            all_x = np.concatenate([X for i in range(int(len(y)/len(X)))])  
#        else:
#            all_x = X
#        coefStim_ = self.coefStim_.reshape((self.coefStim_.shape[0] * self.coefStim_.shape[1], self.coefStim_.shape[2]))
#        
#        if transform_x:
#            return all_x @ coefStim_[:, comp]
#        if transform_y:
#            return y @ self.coefResponse_[:, comp]
        return self.coefResponse_.T @ self.coefStim_.T @ X

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
                    plt.plot(self.xtimes, self.coefStim_[:,dim[d],c],label='CC #%s, dim: %s' % ((c+1), dim[d]))
        else:
            for c in range(5):
                for d in range(len(dim)):
                    plt.plot(self.xtimes, self.coefStim_[:,dim[d],c],label='CC #%s, dim: %s' % ((c+1), dim[d]))
            for c in range(5,n_comp):
                for d in range(len(dim)):
                    plt.plot(self.xtimes, self.coefStim_[:,dim[d],c])
        if self.feat_names_:
            plt.title('Time filter for {:s}'.format(self.feat_names_[0]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Time (s)')
        plt.ylim([-max(np.abs(self.coefStim_[:,dim,:n_comp].flatten())), max(np.abs(self.coefStim_[:,dim,:n_comp].flatten()))])

    def plot_spatial_filter(self, pos, n_comp=1):
        """Plot the topo of the feature requested.
        Parameters
        ----------
        feat_id : int
            Index of the feature requested
        """
        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(self.coefResponse_, pos, n_topos=n_comp, titles=titles)
        # mne.viz.tight_layout()
        plt.tight_layout()


    def plot_corr(self, pos, n_comp=1):
        """Plot the correlation between the EEG component waveform and the EEG channel waveform.
        Parameters
        ----------
        """
        X = np.load(self.tempX_path_+'.npy')
        y = np.load(self.tempy_path_+'.npy')
        if len(y) > len(X):
            all_x = np.concatenate([X for i in range(int(len(y)/len(X)))])  
        else:
            all_x = X
        coefStim_ = self.coefStim_.reshape((self.coefStim_.shape[0] * self.coefStim_.shape[1], self.coefStim_.shape[2]))

        r = np.zeros((64,n_comp))
        for c in range(n_comp):
            eeg_proj = y @ self.coefResponse_[:, c]
            env_proj = all_x @ coefStim_[:, c]
            for i in range(64):
                r[i,c] = np.corrcoef(y[:,i], eeg_proj)[0,1]
            # cc_corr = np.corrcoef(eeg_proj, env_proj)[0,1]

        titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
        topoplot_array(r, pos, n_topos=n_comp, titles=titles)
        # mne.viz.tight_layout()
        plt.tight_layout()

    def plot_activation_map(self, pos, n_comp=1, lag=0):
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
        
        if self.lag_y | self.f_bank_used:
            if self.f_bank_used:
                a_map = np.reshape(a_map,(len(self.f_bank_freqs),self.n_chans_,self.coefResponse_.shape[1]))
            else:
                a_map = np.reshape(a_map,(self.ylags.shape[0],self.n_chans_,self.coefResponse_.shape[1]))
            titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
            fig = plt.figure(figsize=(12, 10), constrained_layout=False)
            outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
            for c in range(n_comp):
                inner_grid = outer_grid[c].subgridspec(1, 1)
                ax = plt.Subplot(fig, inner_grid[0])
                # im, _ = mne.viz.plot_topomap(a_map[lag,:,c], pos, axes=ax, show=False)
                topomap(a_map[lag,:,c], pos, axes=ax, show=False)
                ax.set(title=titles[c])
                fig.add_subplot(ax)
            # mne.viz.tight_layout()
            fig.tight_layout()
            
        else:
            titles = [r"CC #{:d}, $\rho$={:.3f} ".format(k+1, c) for k, c in enumerate(self.score_)]
            topoplot_array(a_map, pos, n_topos=n_comp, titles=titles)
            # mne.viz.tight_layout()
            plt.tight_layout()

    def plot_compact_time(self, n_comp=2, dim=0):
        plt.imshow(self.coefStim_[:, dim, :n_comp].T, aspect='auto', origin='bottom', extent=[self.xtimes[0], self.xtimes[-1], 0, n_comp])
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
                           extent=[self.xtimes[0], self.xtimes[-1], 0, n_dim],
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