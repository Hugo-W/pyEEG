.. role:: hidden
    :class: hidden-section

Simulation module
=================

.. automodule:: pyeeg.simulate
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: pyeeg.simulate

.. autosummary::
   :toctree: generated/
   :template: class.rst

   NeuralMassNode
   NeuralMassNetwork
   HopfOscillator
   Phasor
   WilsonCowan
   Kuramoto
   CTRNN
   JansenRit
   JansenRitExtended
   JRNetwork


Functions
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    simulate_ar
    simulate_var
    simulate_var_from_cov
    linear_coupling
    diffusive_coupling
    kuramoto_coupling
    dummy_trf_kernel
    simulate_smooth_input
    simulate_pulse_inputs
    simulate_trf_output