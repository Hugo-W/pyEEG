# -*- coding: utf-8 -*-
"""
Testing Connectivity measures.
This could be tested using dummy variables and against result from fieldtrip's or mne toolboxes.

Updates:
- 2023/11/06: Do not know when this was created, no proper testing done: starting jackknife_resample and phase_transfer_entropy
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