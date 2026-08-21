"""
Modelling submodule.

Re-exports the public modelling API so that ``from pyeeg.models import ...``
keeps working after the split of the former monolithic :mod:`pyeeg.models`
module into :mod:`pyeeg.models.trf` and :mod:`pyeeg.models.var`.
"""

from .trf import TRFEstimator
from .var import fit_ar, fit_var

__all__ = [
    "TRFEstimator",
    "fit_ar",
    "fit_var",
]