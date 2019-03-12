Tutorials
=========

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

.. toctree::
    :maxdepth: 1
    :caption: Jupyter Notebooks Examples

    examples/TRF_wordonsets
    examples/CCA_envelope
    examples/import_WordVectors
    examples/TRF_syntactic_feats

.. .. include:: ../../examples/TRF_wordonsets.rst
