.. role:: hidden
    :class: hidden-section

Features module
===============

.. automodule:: pyeeg.features
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: pyeeg.features

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SyntacticFeatureExtractor
   AlignmentHandler
   TextGridParser
   FeaturePipeline
   FeatureReducer

.. note::

   :class:`LLMFeatureExtractor` (and its :class:`~pyeeg.features.llm_features.LLMFeatureConfig`)
   live in :mod:`pyeeg.features.llm_features`, which requires `torch`. Install the
   optional ``[features]`` extra (``pip install natmeeg[features]``) to use and
   auto-document the LLM feature extraction classes. They are omitted from the
   autosummary below so the docs build succeeds in environments without torch.

The following dataclasses are defined in the :mod:`pyeeg.features.alignment` submodule:

.. currentmodule:: pyeeg.features.alignment

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Interval
   TextGrid

The following dataclasses are defined in the :mod:`pyeeg.features.pipeline` submodule:

.. currentmodule:: pyeeg.features.pipeline

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FeatureSpec
   PipelineConfig

The following dataclasses are defined in the :mod:`pyeeg.features.reduction` submodule:

.. currentmodule:: pyeeg.features.reduction

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ReductionConfig

The following dataclasses are defined in the :mod:`pyeeg.features.syntactic_features` submodule:

.. currentmodule:: pyeeg.features.syntactic_features

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ParserConfig
