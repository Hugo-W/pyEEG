"""
Feature Reduction

This module provides functionality for reducing the dimensionality of
feature sets using PCA, ICA, and other techniques. The :class:`FeatureReducer`
class wraps the reduction methods and exposes a scikit-learn-like
fit/transform/inverse_transform interface.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .._logging import LOGGER


@dataclass
class ReductionConfig:
    """Configuration for feature reduction.

    Attributes
    ----------
    method : str
        Reduction method to use. One of ``"pca"`` (principal component
        analysis), ``"ica"`` (independent component analysis via whitening),
        or ``"none"`` (identity transformation). Defaults to ``"pca"``.
    n_components : Optional[int], optional
        Number of components to retain. If ``None``, the number is derived
        from ``variance_threshold`` (PCA) or kept equal to the input
        dimensionality (ICA). Defaults to ``None``.
    variance_threshold : float
        For PCA with ``n_components=None``: the cumulative explained-variance
        ratio at which the component count is selected. Defaults to ``0.95``.
    random_state : Optional[int], optional
        Seed for reproducible results. **Currently reserved**: the implemented
        PCA/ICA routines are deterministic and do not use this field.
        Defaults to ``None``.
    whiten : bool
        Whether to whiten the reduced components. **Currently reserved**: the
        implemented routines do not apply an explicit whitening step
        (ICA internally whitens as part of its construction). Defaults to
        ``False``.
    """

    method: str = "pca"
    n_components: Optional[int] = None
    variance_threshold: float = 0.95
    random_state: Optional[int] = None
    whiten: bool = False


class FeatureReducer:
    """Reduce the dimensionality of feature sets.

    Provides a scikit-learn-like interface for reducing 2D feature matrices
    (samples x features) to a lower-dimensional space. The reduction behavior
    is governed by the ``method`` field of the configuration:

    - ``"pca"``: principal component analysis. Components are the principal
      eigenvectors of the feature covariance, sorted by descending
      eigenvalue; the number retained is ``n_components`` or is derived from
      ``variance_threshold``.
    - ``"ica"``: independent component analysis implemented as a whitening
      transform of the centered data. The number retained is ``n_components``
      or the full input dimensionality.
    - ``"none"``: identity transformation; features pass through unchanged.

    Data is centered (mean subtracted per feature) before reduction, and the
    fitted mean is added back by :meth:`inverse_transform`.

    Parameters
    ----------
    config : ReductionConfig
        Configuration of the reduction method and component selection.

    Attributes
    ----------
    config : ReductionConfig
        Configuration of the reducer.
    _fitted : bool
        Whether :meth:`fit` has been called successfully.
    _components : ndarray or None
        Fitted component matrix (rows are components), or ``None`` before fit
        / for ``method='none'``.
    _explained_variance : ndarray or None
        Per-component explained variance ratios (PCA only), or ``None``
        otherwise.
    _mean : ndarray or None
        Per-feature mean used to center the data, or ``None`` before fit.
    _n_components : int or None
        Number of retained components, or ``None`` before fit.
    """

    def __init__(self, config: ReductionConfig):
        """Initialize the reducer with a configuration.

        Parameters
        ----------
        config : ReductionConfig
            Configuration of the reduction method and component selection.
        """
        self.config = config
        self._fitted = False
        self._components: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._n_components: Optional[int] = None
    
    def _validate_input(self, features: np.ndarray) -> None:
        if not isinstance(features, np.ndarray):
            raise ValueError("Features must be a numpy array")
        if features.ndim != 2:
            raise ValueError("Features must be 2D")
        if features.size == 0:
            raise ValueError("Features array is empty")
    
    def _center_data(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None:
            self._mean = np.mean(features, axis=0)
        return features - self._mean
    
    def _uncenter_data(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None:
            return features
        return features + self._mean
    
    def fit(self, features: np.ndarray) -> 'FeatureReducer':
        """Fit the reducer to the features.

        Computes the components and the per-feature mean from ``features``
        according to ``self.config.method``. After fitting, the reducer can
        be used to transform new data.

        Parameters
        ----------
        features : ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        self : FeatureReducer
            The fitted reducer.

        Raises
        ------
        ValueError
            If ``features`` is not a non-empty 2D numpy array, or if
            ``self.config.method`` is not one of ``"pca"``, ``"ica"``, or
            ``"none"``.
        """
        self._validate_input(features)
        
        if self.config.method == 'pca':
            self._fit_pca(features)
        elif self.config.method == 'ica':
            self._fit_ica(features)
        elif self.config.method == 'none':
            self._fitted = True
        else:
            raise ValueError(f"Unknown reduction method: {self.config.method}")
        
        self._fitted = True
        return self
    
    def _fit_pca(self, features: np.ndarray):
        """Fit PCA to the features.

        Centers the data, computes the covariance matrix and its eigendecomposition,
        stores the eigenvectors sorted by descending eigenvalue as components,
        and selects the number of components from ``n_components`` or
        ``variance_threshold``.

        Parameters
        ----------
        features : ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        centered = self._center_data(features)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        total_variance = np.sum(eigenvalues)
        self._explained_variance = eigenvalues / total_variance
        self._components = eigenvectors.T
        
        if self.config.n_components is not None:
            self._n_components = min(self.config.n_components, len(eigenvalues))
        else:
            cumulative_variance = np.cumsum(self._explained_variance)
            self._n_components = np.argmax(cumulative_variance >= self.config.variance_threshold) + 1
    
    def _fit_ica(self, features: np.ndarray):
        """Fit ICA to the features.

        Centers the data and computes a whitening transform from the
        eigendecomposition of the covariance matrix (small eigenvalues are
        floored at ``1e-10``). The whitened data transpose is stored as the
        components, and the number of components is selected from
        ``n_components`` or kept equal to the input dimensionality.

        Parameters
        ----------
        features : ndarray
            2D array of shape ``(n_samples, n_features)``.
        """
        centered = self._center_data(features)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues[eigenvalues < 1e-10] = 1e-10
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        whitened = centered @ whitening_matrix.T
        self._components = whitened.T
        
        if self.config.n_components is not None:
            self._n_components = min(self.config.n_components, whitened.shape[1])
        else:
            self._n_components = whitened.shape[1]
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features to reduced space.

        Projects ``features`` onto the fitted components (after centering with
        the fitted mean). With ``method='none'``, the input is returned
        unchanged.

        Parameters
        ----------
        features : ndarray
            2D array of shape ``(n_samples, n_features)``. The number of
            features must match the fitted dimensionality.

        Returns
        -------
        reduced : ndarray
            2D array of shape ``(n_samples, n_components)`` with the reduced
            features, or the input unchanged for ``method='none'``.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted (call :meth:`fit` first).
        ValueError
            If ``features`` is not a non-empty 2D numpy array, or if
            ``self.config.method`` is not one of ``"pca"``, ``"ica"``, or
            ``"none"``.
        """
        if not self._fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        self._validate_input(features)
        
        if self.config.method == 'pca':
            centered = self._center_data(features)
            return centered @ self._components[:self._n_components].T
        elif self.config.method == 'ica':
            centered = self._center_data(features)
            return centered @ self._components[:self._n_components].T
        elif self.config.method == 'none':
            return features
        else:
            raise ValueError(f"Unknown reduction method: {self.config.method}")
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit the reducer and transform the features.

        Convenience method equivalent to calling :meth:`fit` followed by
        :meth:`transform` on the same data.

        Parameters
        ----------
        features : ndarray
            2D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        reduced : ndarray
            2D array of shape ``(n_samples, n_components)`` with the reduced
            features, or the input unchanged for ``method='none'``.
        """
        self.fit(features)
        return self.transform(features)
    
    def inverse_transform(self, reduced_features: np.ndarray) -> np.ndarray:
        """Transform reduced features back to original space.

        Reconstructs data in the original feature space by projecting the
        reduced features back onto the fitted components and adding back the
        fitted mean. With ``method='none'``, the input is returned unchanged.

        Parameters
        ----------
        reduced_features : ndarray
            2D array of shape ``(n_samples, n_components)`` of reduced
            features.

        Returns
        -------
        features : ndarray
            2D array of shape ``(n_samples, n_features)`` reconstructed in
            the original feature space, or the input unchanged for
            ``method='none'``.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted (call :meth:`fit` first).
        ValueError
            If ``self.config.method`` is not one of ``"pca"``, ``"ica"``, or
            ``"none"``.
        """
        if not self._fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")
        
        if self.config.method == 'pca':
            reconstructed = reduced_features @ self._components[:self._n_components]
            return self._uncenter_data(reconstructed)
        elif self.config.method == 'ica':
            return self._uncenter_data(reduced_features @ self._components[:self._n_components])
        elif self.config.method == 'none':
            return reduced_features
        else:
            raise ValueError(f"Unknown reduction method: {self.config.method}")
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Return the per-component explained variance ratios.

        Returns
        -------
        explained_variance : ndarray or None
            1D array of explained variance ratios for each component (PCA
            only), or ``None`` if not fitted or for non-PCA methods.
        """
        return self._explained_variance

    def get_components(self) -> Optional[np.ndarray]:
        """Return the fitted components.

        Returns
        -------
        components : ndarray or None
            2D array of shape ``(n_components, n_features)`` with the retained
            components (each row is a component), or ``None`` if not fitted or
            no components are available (e.g. ``method='none'``).
        """
        if self._components is not None and self._n_components is not None:
            return self._components[:self._n_components]
        return None

    def get_n_components(self) -> Optional[int]:
        """Return the number of retained components.

        Returns
        -------
        n_components : int or None
            Number of components retained after :meth:`fit`, or ``None`` if
            not fitted.
        """
        return self._n_components