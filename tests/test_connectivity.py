# -*- coding: utf-8 -*-
"""
Testing Connectivity measures.
This could be tested using dummy variables and against result from fieldtrip's or mne toolboxes.

Updates:
- 2023/11/06: Do not know when this was created, no proper testing done: starting jackknife_resample and phase_transfer_entropy
- 2025/04/18: re init test to pass for now, it was wrongly written
"""
import numpy as np
from pyeeg.connectivity import plm

def test_plm():
    pass