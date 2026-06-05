"""
Temporal Response Function (TRF) Estimation

This module provides the TRFEstimator class for estimating Temporal Response
Functions from neural data and stimuli.

The TRFEstimator can now integrate with the feature extraction module to
handle naturalistic stimuli with rich feature representations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TRFConfig:
    """Configuration for TRF estimation."""
    lags: List[float] = field(default_factory=lambda: [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    lambda_reg: float = 1.0
    solver: str = "svd"
    max_iter: int = 1000
    tol: float = 1e-6
    use_features: bool = False
    feature_names: List[str] = field(default_factory=list)
    validate_alignment: bool = True


class TRFEstimator:
    """
    Estimate Temporal Response Functions from neural data and stimuli.
    
    This class can now integrate with the feature extraction module to handle
    naturalistic stimuli. When use_features=True, it expects features to be
    provided instead of raw stimuli, and will align them appropriately.
    
    Args:
        config: TRF estimation configuration
    
    Example:
        >>> from pyeeg.features import StimulusEncoder
        >>> encoder = StimulusEncoder()
        >>> encoder.add_llm_features(['surprisal', 'entropy'])
        >>> encoder.add_syntactic_features(['depth'])
        >>> features, _ = encoder.encode(text, textgrid)
        >>> trf = TRFEstimator(TRFConfig(lags=[-0.1, 0, 0.1, 0.2], use_features=True))
        >>> trf.fit(signal, features)
    """
    
    def __init__(self, config: TRFConfig):
        self.config = config
        self._fitted = False
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._lag_matrix: Optional[np.ndarray] = None
    
    def _validate_inputs(
        self,
        signal: np.ndarray,
        stimulus: Union[np.ndarray, Dict[str, np.ndarray]]
    ):
        """Validate input signal and stimulus."""
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        if signal.ndim != 2:
            raise ValueError("Signal must be 2D (n_samples, n_channels)")
        
        if isinstance(stimulus, np.ndarray):
            if stimulus.ndim != 2:
                raise ValueError("Stimulus must be 2D (n_samples, n_features)")
            if stimulus.shape[0] != signal.shape[0]:
                raise ValueError("Stimulus and signal must have same number of samples")
        elif isinstance(stimulus, dict):
            for name, arr in stimulus.items():
                if not isinstance(arr, np.ndarray):
                    raise ValueError(f"Feature {name} must be a numpy array")
                if arr.ndim != 1 and arr.ndim != 2:
                    raise ValueError(f"Feature {name} must be 1D or 2D")
                if arr.shape[0] != signal.shape[0]:
                    raise ValueError(f"Feature {name} has wrong number of samples")
        else:
            raise ValueError("Stimulus must be numpy array or feature dictionary")
    
    def _create_lag_matrix(
        self,
        stimulus: np.ndarray,
        lags: List[float],
        sampling_rate: float
    ) -> np.ndarray:
        """Create a lag matrix for TRF estimation."""
        n_samples = stimulus.shape[0]
        n_features = stimulus.shape[1]
        n_lags = len(lags)
        
        lag_samples = [int(lag * sampling_rate) for lag in lags]
        lag_matrix = np.zeros((n_samples, n_features * n_lags))
        
        for i, lag in enumerate(lag_samples):
            if lag >= 0:
                lag_matrix[lag:, i * n_features:(i + 1) * n_features] =                     stimulus[:-lag, :]
            else:
                lag_abs = abs(lag)
                lag_matrix[:lag_abs, i * n_features:(i + 1) * n_features] =                     stimulus[lag_abs:, :]
        
        return lag_matrix
    
    def _prepare_features(
        self,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature dictionary for TRF estimation."""
        sorted_names = sorted(features.keys())
        feature_matrix = np.column_stack([features[name] for name in sorted_names])
        return feature_matrix, sorted_names
    
    def fit(
        self,
        signal: np.ndarray,
        stimulus: Union[np.ndarray, Dict[str, np.ndarray]],
        sampling_rate: float = 1000.0
    ):
        """Fit the TRF model."""
        self._validate_inputs(signal, stimulus)
        
        if isinstance(stimulus, dict):
            if not self.config.use_features:
                raise ValueError("Feature dictionary provided but use_features=False")
            stimulus, self._feature_names = self._prepare_features(stimulus)
        
        self._lag_matrix = self._create_lag_matrix(
            stimulus,
            self.config.lags,
            sampling_rate
        )
        
        X = np.column_stack([np.ones(len(self._lag_matrix)), self._lag_matrix])
        y = signal
        
        if self.config.solver == 'svd':
            self._coefficients, self._intercept = self._svd_regress(X, y)
        elif self.config.solver == 'lstsq':
            self._coefficients, self._intercept = self._lstsq_regress(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")
        
        self._fitted = True
        return self
    
    def _svd_regress(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve regression using SVD."""
        intercept_col = X[:, 0:1]
        X_pred = X[:, 1:]
        
        X_centered = X_pred - np.mean(X_pred, axis=0)
        y_centered = y - np.mean(y, axis=0)
        
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        s_reg = s / (s**2 + self.config.lambda_reg)
        X_pinv = Vt.T @ np.diag(s_reg) @ U.T
        
        coefficients = X_pinv @ y_centered
        intercept = np.mean(y, axis=0) - np.mean(X_pred, axis=0) @ coefficients
        
        return coefficients.T, intercept
    
    def _lstsq_regress(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve regression using least squares."""
        intercept_col = X[:, 0:1]
        X_pred = X[:, 1:]
        
        n_predictors = X_pred.shape[1]
        X_reg = np.vstack([X_pred, np.sqrt(self.config.lambda_reg) * np.eye(n_predictors)])
        y_reg = np.vstack([y, np.zeros((n_predictors, y.shape[1]))])
        
        coefficients, residuals, rank, s = np.linalg.lstsq(X_reg, y_reg, rcond=None)
        intercept = np.mean(y, axis=0) - np.mean(X_pred, axis=0) @ coefficients.T
        
        return coefficients.T, intercept
    
    def predict(self, stimulus: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """Predict neural response from stimulus."""
        if not self._fitted:
            raise RuntimeError("TRFEstimator not fitted. Call fit() first.")
        
        if isinstance(stimulus, dict):
            if not self.config.use_features:
                raise ValueError("Feature dictionary provided but use_features=False")
            stimulus, _ = self._prepare_features(stimulus)
        
        lag_matrix = self._create_lag_matrix(
            stimulus,
            self.config.lags,
            1000.0
        )
        
        X = np.column_stack([np.ones(len(lag_matrix)), lag_matrix])
        intercept_col = X[:, 0:1]
        X_pred = X[:, 1:]
        
        prediction = X_pred @ self._coefficients.T + self._intercept
        
        return prediction
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        return self._coefficients
    
    def get_intercept(self) -> Optional[np.ndarray]:
        return self._intercept
    
    def get_feature_names(self) -> List[str]:
        return self._feature_names
    
    def score(
        self,
        signal: np.ndarray,
        stimulus: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> float:
        """Calculate the R^2 score for the model."""
        prediction = self.predict(stimulus)
        ss_res = np.sum(np.square(signal - prediction))
        ss_tot = np.sum(np.square(signal - np.mean(signal, axis=0)))
        return 1 - ss_res / ss_tot