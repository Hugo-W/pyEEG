"""
natMEEG package for analyzing M/EEG data with naturalistic stimuli.

This library provides tools for processing M/EEG (Magnetoencephalography and Electroencephalography)
data, particularly for experiments using naturalistic stimuli such as continuous speech,
music, or other complex, real-world inputs. It supports analysis of continuous M/EEG data
and generation of temporal response functions (TRFs) from continuous signals or
real-valued events (e.g., word-level or phoneme-level features).

The package is built on top of MNE-Python and scikit-learn.

Main Classes:
    TRFEstimator: Temporal Response Function estimation
    CCA_Estimator: Canonical Correlation Analysis
    MultichanWienerFilter: Multi-channel Wiener filtering
    Whitener: Data whitening
    mCCA: Multi-way CCA (also known as hyperalignment)

Usage:
    import pyeeg
    from pyeeg import TRFEstimator

    trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=fs, alpha=100.0)
    trf.fit(X, y)  # X: stimulus, y: M/EEG signal

See README.md for more details and examples.

2019-2026, Hugo Weissbart <hugo.weissbart@donders.ru.nl>
"""

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s - %(levelname)s - %(message)s",
)

# This enables access to all submodules from the top-level
# pyeeg module
from ._logging import set_log_level, get_logger
from . import connectivity, io, models, preprocess, vizu, utils, simulate, features
from .models import TRFEstimator
from .cca import CCA_Estimator
from .preprocess import MultichanWienerFilter, Whitener
from .mcca import mCCA
from .features import (
    LLMFeatureExtractor,
    SyntacticFeatureExtractor,
    AlignmentHandler,
    TextGridParser,
    FeaturePipeline,
    FeatureReducer,
)
from .version import __version__

# Public API
__all__ = [
    # Classes
    'TRFEstimator',
    'AlignmentHandler',
    'FeaturePipeline',
    'CCA_Estimator',
    'MultichanWienerFilter',
    'Whitener',
    'mCCA',
    'LLMFeatureExtractor',
    'SyntacticFeatureExtractor',
    'TextGridParser',
    'FeatureReducer',
    # Logging
    'set_log_level',
    'get_logger',
    # Submodules
    'connectivity',
    'io',
    'models',
    'preprocess',
    'vizu',
    'utils',
    'simulate',
    'features',
    # Version
    '__version__',
]
