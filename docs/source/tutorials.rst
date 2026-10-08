Tutorials
=========

The notebooks below are self-contained and use simulated signals, so they can
be run without an EEG recording or private stimulus files. Figures generated
by the notebook cells are rendered directly in the HTML documentation.

TRF tutorials
-------------

Two complementary tutorials cover Temporal Response Function (TRF) estimation.
Both are self-contained and use simulated signals.

.. toctree::
    :maxdepth: 1

    examples/TRF_simulation_tutorial
    examples/TRF_advanced_tutorial

- **A gentle introduction to TRFs** (`TRF_simulation_tutorial`) — a minimal
  walkthrough: simulate two stimulus features, fit the simplest TRF, and
  visualise the result. Best starting point.
- **TRF estimation: inference, regularisation, and solvers**
  (`TRF_advanced_tutorial`) — an in-depth tutorial covering statistical
  inference (permutation tests, bootstrap CIs), banded regularisation,
  cross-validation, robust fitting, solver comparison, and multi-channel TRFs.

Loading Word-level features
---------------------------

.. code-block:: python

    from pyeeg.io import AlignedSpeech
    speech = AlignedSpeech(onset=0., srate=125, path_audio='path/to/audio/file')

Loading Word vectors
--------------------

.. code-block:: python

    from pyeeg.io import WordLevelFeatures
    wlf = wfeats = WordLevelFeatures(path_praat_env=dur_path, path_wordonsets=surp_path, path_surprisal=surp_path)

Loading processed EEG (processed by EEGLAB)
-------------------------------------------

To load EEG data from ``.set`` files, the code would be:

.. code-block:: python

    from pyeeg.io import eeglab2mne
    eeg = eeglab2mne(filepath)

Older, data-dependent examples
--------------------------------

These examples are retained for now. They require local EEG/audio files and
may need adaptation to current APIs.

.. toctree::
    :maxdepth: 1

    examples/TRF_wordonsets
    examples/CCA_envelope
    examples/import_WordVectors
    examples/TRF_syntactic_feats

.. .. include:: ../../examples/TRF_wordonsets.rst
