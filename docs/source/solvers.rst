.. role:: hidden
    :class: hidden-section

Solvers module
==============

.. automodule:: pyeeg.solvers
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: pyeeg.solvers

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Solver
   SolverResult
   SVDSolver
   LSTSQSolver
   ConjugateGradientSolver
   IRLSSolver
   ScipyRobustSolver


Functions
---------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    create_laplacian_matrix
    create_quadratic_regularizer
    svd_solver
    incomplete_cholesky_preconditioner
    diagonal_preconditioner
    conjugate_gradient
    block_conjugate_gradient