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

    lag_matrix
    lag_span
    lag_sparse
    rolling_func
    moving_average
    chunk_data
    shift_array

Signal
''''''

Signal processing related.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    signal_envelope
    fir_order
    find_knee_point
    is_pos_def

Other
'''''

Miscelaneous, e.g. system related, or characterisation...

.. autosummary::
    :toctree: generated/
    :template: function.rst

    mem_check
    _is_1d
