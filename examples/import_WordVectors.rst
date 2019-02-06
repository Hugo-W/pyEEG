
Work with WordVectors
=====================

This notebook will show how to import word vectors into the
``WordLevelFeatures`` class.

.. code:: ipython3

    import os
    from pyeeg.io import WordLevelFeatures

.. code:: ipython3

    story_id = 0
    
    # Path to data ocntaining onsets of words and duration of speech segments
    env_path = '/media/hw2512/SeagateExpansionDrive/EEG_data/Katerina_experiment/story_parts/alignement_data/'
    wordfreq_path = '/media/hw2512/SeagateExpansionDrive/EEG_data/Katerina_experiment/story_parts/word_frequencies/'
    list_wordfreq_files = [item for item in os.listdir(wordfreq_path) if item.endswith('timed.csv')]
    list_stories = [item.strip('_word_freq_timed.csv') for item in list_wordfreq_files]
    list_env_files = [os.path.join(env_path, s, s + '_125Hz.Env') for s in list_stories]
    
    # Path to WordVectors
    wv_path = '/home/hw2512/MachineLearning/Word2Vec/GloVe-1.2/vectors_correct_header.txt'
    
    # Loading word onset and duration for one story:
    wo_path = os.path.join(wordfreq_path, list_wordfreq_files[story_id])
    duration_path = os.path.join(env_path, list_env_files[story_id])

.. code:: ipython3

    wf = WordLevelFeatures(path_praat_env=duration_path, path_wordonsets=wo_path, path_wordvectors=wv_path)


.. parsed-literal::

    INFO:pyeeg.io:Word flushington not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word bashfulest not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word adulation not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word blush not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word blush not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word paucity not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word fellow-men not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word fastened not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word flushington not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word scraggy not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word timid not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word deprecating not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word uninteresting not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word impervious not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word aimless not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word superstitiously not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word monosyllabic not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word laboriously not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word flushington not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word gyp not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word bedmaker not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word unaccountably not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word sluices not in word embedding model; will use rdm instead
    INFO:pyeeg.io:Word flushington not in word embedding model; will use rdm instead


.. code:: ipython3

    feat_matrix = wf.align_word_features(features=['wordvectors'], srate=100, wordonset_feature=True)
    print("First few samples of first 4 dimensions:\n")
    print(feat_matrix[250:350, :4])


.. parsed-literal::

    First few samples of first 4 dimensions:
    
    [[ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 1.          0.07879453  0.09271022 -0.1146835 ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 1.         -0.74636799  0.196638    0.490275  ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.        ]]

