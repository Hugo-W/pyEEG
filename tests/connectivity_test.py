# -*- coding: utf-8 -*-
"""
Testing Connectivity measures.
This could be tested using dummy variables and against result from fieldtrip's or mne toolboxes.
"""
import numpy as np
from pyeeg.connectivity import plm

def plm_test():
    fs = 100
    T = 10
    N = int(fs*T)
    x = np.random.randn(N)
    y = np.random.randn(N)
    c = plm(x, y)
    assert np.allclose(c, 0)