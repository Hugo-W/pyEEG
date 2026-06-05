"""
pyEEG package for analyszing EEG with speech and word-level features.

 import pyeeg.* and have fun decoding!

 2019, Hugo Weissbart
"""

# This enables access to all submodules from the top-level
# pyeeg module
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
    StimulusEncoder
)
from .version import __version__