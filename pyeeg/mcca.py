# -*- coding: utf-8 -*-
"""
Module implementing Multiway-CCA, for preprocessing.

Multiway canonical correlation analysis (MCCA) finds a set of projections
that align several datasets (e.g. EEG data from different subjects) into
a common space where they share a maximal common response. The resulting
summary components can be used to denoise each individual dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

class mCCA(BaseEstimator):
    """
    Class to support mCCA computation on a set of data matrices.
    Typically N data matrices, with SAME number of samples (observations) but possibly
    different number of channels (features).
    A typical use case would be to find common source of activity within each matrix, for instance
    where they carry EEG data for individual subject, and one wants to denoise the EEG data by projecting
    each subject's EEG into a space where they share a common response. The projection matrices will be per
    individual.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep in the whitening PCA applied to each
        dataset. Defaults to ``None`` (all components are kept).

    Attributes
    ----------
    n_datasets_ : int
        Number of datasets that were fitted.
    SCs_ : ndarray (nsamples x n_components)
        Summary components: the principal components of the concatenated
        whitened datasets, i.e. the shared common response across datasets.
    SC_variances : ndarray
        Explained variance of each summary component.
    individual_transforms_ : list of ndarray
        Per-dataset projection matrices mapping each dataset into the shared
        canonical space. One matrix per fitted dataset.

    Methods
    -------
    fit
    canonical_correlate_single
    denoise

    References
    ----------
    De Cheveigné et. al, MCCA of brain signals, 2018, biorXiv
    """
    def __init__(self, n_components=None):
        """Initialize the mCCA estimator.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep in the whitening PCA applied to
            each dataset. If ``None`` (default), all components are kept.
        """
        self.n_components = n_components

    def fit(self, X):
        """Fit the mCCA model on a set of datasets.

        Each dataset is first whitened with a PCA, then a second PCA is
        applied to the concatenation of the whitened datasets to extract the
        shared summary components. Individual projection matrices are
        computed for every dataset.

        Parameters
        ----------
        X : list of array-like (Time x channels) or array-like (subj x time x channels)
            The datasets to align. All datasets must have the same number of
            samples (observations) but may have different numbers of channels.

        Returns
        -------
        self : mCCA
            The fitted estimator. In-place, sets ``n_datasets_``, ``SCs_``,
            ``SC_variances`` and ``individual_transforms_``.

        :no-index:
        """
        self.n_datasets_ = len(X)

        pca1 = []
        X_whiten = []
        print("First PCAs for whitening individual datasets...")
        for k, x in enumerate(X):
            print("Whitening dataset {:d}".format(k+1))
            pca = PCA(whiten=True, n_components=self.n_components)
            X_whiten.append(pca.fit_transform(x))
            pca1.append((pca.components_, pca.explained_variance_))

        print("Second PCA on concatenated whitened datasets...")
        pca = PCA()
        Y = pca.fit_transform(np.concatenate(X_whiten, axis=1))
        self.SCs_ = Y
        self.SC_variances = pca.explained_variance_

        print("Computing individual transform matrices...")
        self.individual_transforms_ = []
        D = 0
        for k, d in enumerate([x.shape[1] for x in X_whiten]):
            #sigma = np.diag(np.sqrt(pca1[k][1])**(-1))
            V_pca1 = pca1[k][0] / np.sqrt(pca1[k][1][:, np.newaxis])
            #V_pca2 = pca.components_.T[D:D + d].T # d rows of projection matrix for Y
            V_pca2 = pca.components_[:, D:D + d] # d rows of projection matrix for Y
            D += d
            #V = np.dot(V_pca2, np.dot(sigma, V_pca1))
            V = np.dot(V_pca2, V_pca1).T
            self.individual_transforms_.append(V)

    def canonical_correlate_single(self, X, idx):
        """Project one single dataset into its canonical correlate components.

        Parameters
        ----------
        X : array-like (Time x channels)
            The dataset to project.
        idx : int
            Index of the dataset in the fitted list, used to select the
            matching per-dataset transform.

        Returns
        -------
        Y : ndarray
            The dataset projected onto the shared canonical space.

        :no-index:
        """
        return np.dot(X, self.individual_transforms_[idx].T)

    def plot_summary_components_variance(self, normalize=False, axis=None):
        """Plot the variance of each summary component.

        Parameters
        ----------
        normalize : bool
            If ``True``, display the variance as a percentage of datasets
            which share this canonical component. Defaults to ``False``.
        axis : matplotlib.axes.Axes, optional
            Axis on which to plot. If ``None`` (default), a new figure is
            created.

        Returns
        -------
        axis : matplotlib.axes.Axes
            The axis used for plotting.
        """
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        if normalize: # percentage of datasets which share this CC
            axis.stem(self.SC_variances / self.n_datasets_ * 100)
        else:
            axis.stem(self.SC_variances)
        axis.set_xlabel('Summary Comp. #')
        axis.set_ylabel('Variance')

    def denoise(self, X, num_comps, idx):
        """Denoise a dataset by projecting onto its low-rank canonical space.

        The dataset is projected onto the first ``num_comps`` canonical
        components of its own transform, then back into the original
        channel space, retaining the shared (signal) part while discarding
        the components that are not shared across datasets.

        Parameters
        ----------
        X : array-like (Time x channels)
            The dataset to denoise.
        num_comps : int
            Number of canonical components kept for reconstruction.
        idx : int
            Index of the dataset in the fitted list, used to select the
            matching per-dataset transform.

        Returns
        -------
        X_denoised : ndarray
            The denoised dataset, same shape as ``X``.

        :no-index:
        """
        #D = np.dot(self.individual_transforms_[idx][:,:num_comps], pinv(self.individual_transforms_[idx])[:num_comps])
        #proj = self.individual_transforms_[idx][:num_comps].T
        #inv_proj = pinv(self.individual_transforms_[idx])[:, :num_comps].T
        proj = self.individual_transforms_[idx][:, :num_comps]
        inv_proj = np.linalg.pinv(self.individual_transforms_[idx])[:num_comps]
        D = np.dot(proj, inv_proj)
        return np.dot(X, D)