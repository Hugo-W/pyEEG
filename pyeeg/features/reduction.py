"""
Feature Reduction

This module provides functionality for reducing the dimensionality of
feature sets using PCA, ICA, and other techniques.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReductionConfig:
    """Configuration for feature reduction."""
    method: str = "pca"
    n_components: Optional[int] = None
    variance_threshold: float = 0.95
    random_state: Optional[int] = None
    whiten: bool = False


class FeatureReducer:
    """Reduce the dimensionality of feature sets."""
    
    def __init__(self, config: ReductionConfig):
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
        """Fit the reducer to the features."""
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
        """Fit PCA to the features."""
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
        """Fit ICA to the features."""
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
        """Transform features to reduced space."""
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
        """Fit the reducer and transform the features."""
        self.fit(features)
        return self.transform(features)
    
    def inverse_transform(self, reduced_features: np.ndarray) -> np.ndarray:
        """Transform reduced features back to original space."""
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
        return self._explained_variance
    
    def get_components(self) -> Optional[np.ndarray]:
        if self._components is not None and self._n_components is not None:
            return self._components[:self._n_components]
        return None
    
    def get_n_components(self) -> Optional[int]:
        return self._n_components