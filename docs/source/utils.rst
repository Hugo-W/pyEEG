.. role:: hidden
    :class: hidden-section

Utilities
=========

.. automodule:: pyeeg.utils
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: pyeeg.utils

.. autosummary::
   :toctree: generated/
   :template: class.rst

Functions
---------

Array manipulation
''''''''''''''''''

Utils functions to create shifted version of an array, rolling or moving views along axis, etc.

.. autosummary:: 
    :toctree: generated/
    :template: function.rst

    pyeeg.utils.lag_matrix
    pyeeg.utils.lag_span
    pyeeg.utils.lag_sparse
    pyeeg.utils.rolling_func
    pyeeg.utils.moving_average
    pyeeg.utils.chunk_data
    pyeeg.utils.shift_array

Signal
''''''

Signal processing related.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    pyeeg.utils.signal_envelope
    pyeeg.utils.fir_order
    pyeeg.utils.find_knee_point
    pyeeg.utils.is_pos_def

Other
'''''

Miscelaneous, e.g. system related, or characterisation...

.. autosummary::
    :toctree: generated/
    :template: function.rst

    pyeeg.utils.mem_check
    pyeeg.utils._is_1d
