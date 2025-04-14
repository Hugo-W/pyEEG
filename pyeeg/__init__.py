"""
pyEEG package for analyszing EEG with speech and word-level features.

 import pyeeg.* and have fun decoding!

 2019, Hugo Weissbart
"""
# Python 2/3 compatibility (obsolete as of 2020, removing)
# from __future__ import division, print_function, absolute_import

# This enables access to all submodules from the top-level `pyeeg` module
from . import connectivity, io, models, preprocess, vizu, utils, simulate
from .models import TRFEstimator
from .cca import CCA_Estimator
from .preprocess import MultichanWienerFilter, Whitener
from .mcca import mCCA
from .version import __version__