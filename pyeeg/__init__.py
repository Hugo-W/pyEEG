"""
pyEEG package for analyszing EEG with speech and word-level features.

 import pyeeg.* and have fun decoding!

 2019, Hugo Weissbart
"""
from __future__ import division, print_function, absolute_import
from pkg_resources import get_distribution

# This enables access to all submodules from the top-level `pyeeg` module
from . import cca, connectivity, io, mcca, models, preprocess, vizu, utils, simulate
from .version import __version__