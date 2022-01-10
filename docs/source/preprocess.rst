.. role:: hidden
    :class: hidden-section

Preprocessing module
====================

.. automodule:: pyeeg.preprocess
    :no-members:
    :no-inherited-members:

Classes
-------

From another module, we can use the multiway CCA as a preprocessing step. Also known as hyperalignment in the literature.

.. currentmodule:: pyeeg.mcca

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pyeeg.mcca.mCCA

Directly in :mod:`pyeeg.preprocess` one can find also the two following classes:

.. currentmodule:: pyeeg.preprocess

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pyeeg.preprocess.MultichanWienerFilter

.. autosummary::
    :toctree: generated/
    :template: class.rst

    WaveletTransform

Summary 
-------

.. currentmodule:: pyeeg.preprocess


.. autosummary:: 
    :toctree: generated/
    :template: function.rst

    pyeeg.preprocess.create_filterbank
    pyeeg.preprocess.apply_filterbank
    pyeeg.preprocess.get_power
    pyeeg.preprocess.covariance
    pyeeg.preprocess.covariances
    pyeeg.preprocess.covariances_extended

Listing of all classes and functions
-------------------------------------

.. automodule:: pyeeg.preprocess
    :members:
    :noindex:

