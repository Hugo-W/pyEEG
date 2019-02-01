#! python
# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Input/Output
~~~~~~~~~~~~~~~

Description
==============

Input/output of data from files:
     - import EEG data from .set files or raw files
     - import features (word onsets, envelopes, etc.)

Module to correctly extract Matlab files,
especially EEGLAB .set files no matter if the version is >=7.3 and
take the data to MNE format.

*Author: Hugo Weissbart*

"""
import logging
import numpy as np
import pandas as pd
import h5py # for mat file version >= 7.3
import mne
from mne.preprocessing.ica import ICA
#import os
#os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
from scipy.io import loadmat

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

try:
    import gensim
    GENSIM_IMPORTED = True
except ImportError:
    LOGGER.critical("Can't import Gensim (is it installed?) Will not load word vectors.")
    GENSIM_IMPORTED = False

def load_mat(fname):
    """
    Loading simple mat file into dictionnary

    Parameters
    ----------
    fname : str
        Absolute path of file to be loaded

    Returns
    -------
    data : dict
        Dictionnary containing variables from .mat file
        (each variables is a key/value pair)
    """
    try:
        data = loadmat(fname)
    except NotImplementedError:
        print(".mat file is from matlab v7.3 or higher, will use HDF5 format.")
        with h5py.File(fname, mode='r') as fid:
            data = {}
            for k, val in fid.iteritems():
                data[k] = val.value
    return data


def _load_eeglab_data(fname):
    """"
    Load eeglab structure and returns data, channel names,
    sampling frequency and other useful data...
    """
    with h5py.File(fname, mode='r') as fid:
        eeg = fid['EEG/data'].value
        srateate = fid['EEG/srateate'].value

        time = fid['EEG/times'].value
        n_events = fid['EEG/event/latency'].shape[0]

        events = np.zeros((n_events,))
        event_type = []
        for k in range(n_events):
            events[k] = fid[fid['EEG/event/latency'][k][0]].value
            event_type.append(''.join([chr(i) for i in fid[fid['EEG/event/type'][k][0]]]))

        chnames = []
        for k in range(fid['EEG/chanlocs/labels'].shape[0]):
            chnames.append(''.join([chr(i) for i in fid[fid['EEG/chanlocs/labels'][k][0]]]))

        return eeg, srateate, time, events, event_type, chnames

def load_ica_matrices(fname):
    """Load ICA matrices from EEGLAB structure .set file

    Parameters
    ----------
    fname : str
        File path to .set EEGLAB file containing ICA matrices

    Returns
    -------
    weights : ndarray (ncomp x nchan)
        Unmixing matrix (form _whitened_ observed data to sources)
    weights : ndarray (nchan x ncomp)
        Mixing matrix (form sources to _whitened_ observed data)
    sphere : ndarray (nchan x nchan)
        Sphering matrix

    Raises
    ------
    IOError
        If ICA matrices are not present in the given .set file

    """
    try:
        eeg = loadmat(fname)['EEG']
        idx_weights = np.where(np.asarray(eeg.dtype.names) == 'icaweights')
        weights = eeg[0, 0][idx_weights]
        if weights is None:
            raise  IOError("ICA matrices not present in file!")
        winv = eeg[0, 0][17]
        sphere = eeg[0, 0][18]

    except NotImplementedError:
        LOGGER.warning('Must use hdf5 extraction of .set file data')
        #print("WARNING: must use hdf5 extraction of .set file data")
        with h5py.File(fname, mode='r') as fid:
            weights = fid['EEG/icaweights'].value
            winv = fid['EEG/icawinv'].value
            sphere = fid['EEG/icasphere'].value

    return weights, winv, sphere

def eeglab2mne(fname, montage='standard_1020', event_id=None, load_ica=False):
    """Get an EEGLAB dataset into a MNE Raw object.

    Parameters
    ----------
    input_fname : str
        Path to the .set file. If the data is stored in a separate .fdt file,
        it is expected to be in the same folder as the .set file.
    montage : str | None | instance of montage
        Path or instance of montage containing electrode positions.
        If None, sensor locations are (0,0,0). See the documentation of
        :func:`mne.channels.read_montage` for more information.
    event_id : dict
        Map event id (integer) to string name of corresponding event.
        This allows to smoothly load EEGLAB event structure when overlapping
        events (e.g. boundaries) occur.
    load_ica : bool
        Default to False. Load ica matrices from eeglab structure if available
        and attempt to transfer them into the ica structure of MNE.

    Returns
    -------
    raw : Instance of RawEEGLAB
        A Raw object containing EEGLAB .set data.
    ica : Instance of ICA
        If load_ica True

    Note
    ----
    ICA matrices in ICA MNE object might not entirely capture the decomposition.
    To apply projections (i.e. remove some components from observed EEG data) it
    might be better to load directly the matrices and do it by hand, where:
        - icawinv = pinv(icaweights * icasphere)
        - ica_act = icaweights * icasphere * eegdata
    See for references:
        - https://benediktehinger.de/blog/science/ica-weights-and-invweights/
        - MNE PR: https://github.com/mne-tools/mne-python/pull/5114/files

    """
    montage_mne = mne.channels.montage.read_montage(montage)

    try:
        raw = mne.io.read_raw_eeglab(input_fname=fname, montage=montage_mne, event_id=event_id, preload=True)
    except NotImplementedError:
        print("Version 7.3 matlab file detected, will load 'by hand'")
        eeg, srateate, _, _, _, ch_names = _load_eeglab_data(fname)
        info = mne.create_info(ch_names=ch_names, sfreq=srateate, ch_types='eeg', montage=montage_mne)
        raw = mne.io.RawArray(eeg.T, info)

    if load_ica:
        weights, winv, sphere = load_ica_matrices(fname)
        ica = ICA(n_components=winv.shape[1], max_pca_components=winv.shape[1])
        ica.fit(raw, decim=2, start=1., stop=60.)
        ica.unmixing_matrix_ = weights.dot(sphere.dot(ica.pca_components_.T))
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        return raw, ica
    return raw

def extract_duration_praat(fname):
    "Extract value of speech segment duration from Praat file."
    with open(fname, 'r') as fid:
        headers = fid.readlines(80)
        duration = float(headers[4])
        return duration

def load_surprisal_values(filepath, eps=1e-12):
    "Load surprisal values from the given file."
    if filepath is None:
        return None
    dataframe = pd.read_csv(filepath, delim_whitespace=True, usecols=[0, 1, 2, 3, 4])
    surprisal = dataframe.loc[np.logical_and(dataframe.Word != '</s>', dataframe.Word != '\''), 'P(NET)'].get_values()
    return -np.log(surprisal + eps)

def load_wordfreq_values(filepath, unkval=111773390, normfactor=3.2137e12):
    "Load word frequency, and returns -log of it (scaled by median value)"
    csv = pd.read_csv(filepath)
    wordfreq = csv.frequency.get_values()
    # Replace unknown by (global, form my stories) median value:
    wordfreq[wordfreq == -1] = unkval
    # Transform into log proba, normalized by maximum count, roughly equals to total count:
    wordfreq = -np.log(wordfreq/normfactor)
    return wordfreq

def get_word_onsets(filepath):
    """Simply load word lists and respective onsets for a given sound file/story parts...

    Parameters
    ----------
    filepath : str
        path to .csv file that contains entry for onset/offset time of words along with
        word string themselves

    Returns
    -------
    list
        Onset time of each words
    list
        Actual word list

    Note
    ----
    Here we expect the file to be a comma separayted file with a named column for *onset* and for
    *word* respectively.

    """
    csv = pd.read_csv(filepath)
    return csv.word.get_values(), csv.onset.get_values()

class WordLevelFeatures:
    """Gather word-level linguistic features based on old and rigid files, generated from
    various sources (RNNLM, custom scripts for word frequencies, Forced-Alignment toolkit, ...)

    Parameters
    ----------
    path_praat_env : str
        Path to Praat file for envelopes (.Env files in
        `Katerina_experiment/story-parts/alignment_data/`)
    path_surprisal : str
        Path to file containing surprisal values, as extracted with RNNLM toolkit.
    path_wordvectors : str
        Path to Word Vector file, can be binary or textual. Though for text, must start
        with a line indicating number of words and dimension (i.e. `gensim` compatible)
    path_wordfrequency : str
        Path to file containing word counts as exctracted from Google Unigrams.
    path_wordonsets : str
        *Important!* Path to file with word onsets. For now, this must be one of the `*_timed.csv`
        files, for instance as in `Katerina_experiment/story-parts/wordfrequencies/*`
    path_transcript : str
        Path to actual text data of corresponding speech segment (transcript).
    path_audio : str
        Path to audio file
    rnnlm_model : str
        Path to RNNLM toolkit to compute surprisal and entropy values from a text.


    Attributes
    ----------
    duration : float
        Duration in seconds of the speech segment represented by this instance
    surprisal : array-like
        Surprisal values for each words
    wordfrequency : array-like
        Word frequency values for each words
    wordonsets : array-like
        Onsets values for each words
    wordlist : list[str]
        List of words from which each feature is being measured
    wordvectors : array-like (nwords x ndimension)
        Matric of word vectors, i.e. concatenation of word vector correpsonding to words in the text
    vectordim : int
        Number of dimensions in the word embedding currently loaded.


    Examples
    --------
    >>> from pyeeg.io import WordLevelFeatures
    >>> import os
    >>> from os.path import join
    >>> wordfreq_path = '/media/hw2512/SeagateExpansionDrive/EEG_data/Katerina_experiment/story_parts/word_frequencies/'
    >>> env_path = '/media/hw2512/SeagateExpansionDrive/EEG_data/Katerina_experiment/story_parts/alignement_data/'
    >>> surprisal_path = '/media/hw2512/SeagateExpansionDrive/EEG_data/Katerina_experiment/story_parts/surprisal/'
    >>> list_wordfreq_files = [item for item in os.listdir(wordfreq_path) if item.endswith('timed.csv')]
    >>> list_surprisal_files = [item for item in os.listdir(surprisal_path) if item.endswith('3.txt')]
    >>> list_stories = [item.strip('_word_freq_timed.csv') for item in list_wordfreq_files]
    >>> list_env_files = [os.path.join(env_path, s, s + '_125Hz.Env') for s in list_stories]
    >>> surp_path = os.path.join(surprisal_path, list_surprisal_files[1])
    >>> wf_path = os.path.join(wordfreq_path, list_wordfreq_files[1])
    >>> dur_path = os.path.join(env_path, list_env_files[1])
    >>> wo_path = os.path.join(wordfreq_path, list_wordfreq_files[1])
    >>> wfeats = WordLevelFeatures(path_praat_env=dur_path, path_wordonsets=wo_path, path_surprisal=surp_path, path_wordfrequency=wf_path)
    >>> wf
       surprisal   wordfreq      words     onsets
    0   6.202895   9.631348         if   9.631348
    1  11.555537   6.981747        the   6.981747
    2  25.527839  13.521181  afternoon  13.521181
    3  27.631021   8.300234        was   8.300234
    4   9.720567  12.214926       fine  12.214926
    Duration: 159.08
    >>> feature_matrix = wf.align_word_features(srate=125, features=['surprisal', 'wordfrequency'])

    Note
    ----
    For now, the class assumes that the data are stored in a given format that depended
    on the processing done for the work on surprisal. However it ought to be extended to
    be bale to generate surprisal/word frequency/word onsets/word vectors for a given audio
    files along with its transcript.

    TODO
    ----
    - Extend the class to create word-feature of choices on-the-fly
    - Extract entropy!
    - Be able to extract word onsets from TextGrid data?

    """

    def __init__(self, path_praat_env=None, path_surprisal=None, path_wordvectors=None,
                 path_wordfrequency=None, path_wordonsets=None, path_transcript=None,
                 rnnlm_model=None, path_audio=None):

        if path_praat_env is not None:
            self.duration = extract_duration_praat(path_praat_env)

        try:
            self.wordlist, self.wordonsets = get_word_onsets(path_wordonsets)
        except ValueError:
            raise ValueError("Please supply a path at least for word onsets!")
        self.surprisal = load_surprisal_values(path_surprisal)
        self.wordfrequency = load_wordfreq_values(path_wordfrequency)

        # TODO: load audio and transcript, run rnnlm from here

        self.audio = path_audio
        self.rnnlm = rnnlm_model
        self.transcript = None
        if path_transcript is not None:
            with open(path_transcript, 'r') as fid:
                raw = fid.read()
            self.transcript = raw

        self.wordvectors = None
        self.vectordim = 0
        if GENSIM_IMPORTED and path_wordvectors is not None:
            self.wordvectors = gensim.models.KeyedVectors.load_word2vec_format(path_wordvectors, binary=False, limit=30000).syn0
            self.vectordim = self.wordvectors.shape[1]

    def summary(self):
        "Print a short summary for each variables contained in the instance"
        dataframe = pd.DataFrame(dict(surprisal=self.surprisal, wordfreq=self.wordfrequency,
                                      words=self.wordlist, onsets=self.wordfrequency))

        print(dataframe.head())
        print("\nSummary statistics:")
        print(dataframe.describe())

    def _pprint(self):
        "Pretty print for current object"
        print(self.__repr__)

    def __repr__(self):
        dataframe = pd.DataFrame(dict(surprisal=self.surprisal, wordfreq=self.wordfrequency,
                                      words=self.wordlist, onsets=self.wordonsets))
        short_repr = dataframe.head().__repr__()
        other_repr = "\nDuration: {:.2f}".format(self.duration)
        if self.vectordim > 0:
            other_repr += "\nWord vectors loaded (ndim: {:d})".format(self.vectordim)
        return short_repr + other_repr


    def align_word_features(self, srate, wordonset_feature=True, features=('surprisal',), zscore_fun=None, custom_wordfeats=None):
        """Check that we have the correct number of elements for each features
        Will output a time series with spike for each word features at times from :attr:`self.wordonsets`.
        The duration of the output signal is determined from :attr:`self.duration`.
        Sampling rate needs to be given as a parameter to be able to create the spike time-series.

        Parameters
        ----------
        srate : float
            Sampling rate at which to compute spike-like features
        wordonset_feature : boolean
            Whether to use word onset (always "on" feature) in the
            feature array (Default: True).
        features : list
            List of feature names to be used. These feature name must correspond to
            an attribute of the :class:`WordLevelFeatures` class.
        zscore_fun : function or sklearn.preprocessing.StandardScaler-like instance
            Whether to scale and center the word feature values before
            assigning them to onset sample-time. If method (function) is given, will
            simply apply the function to the data on the fly. If StandardScaler-like
            instance is given, then we assume the object has been fitted beforehand to
            the data and contains mean and variance, thus will use the 'transform'
            method of the instance to scale data. If None, leave as it is. (Default: None)
        custom_wordfeats : ndarray-like (nfeat x nwords)
            Other word-level features that are not embeded in the class instance.
            The only requirement is that the number of words must be the same as
            for the list of word onsets.

        Returns
        -------
        feat : ndarray (ntimes, nfeat)
            Array of spikes, with the word feature value at word onsets.
            ntimes ~ speech_duration * srate

        Raises
        ------
        AssertionError
            If the duration of speech part has not been extracted or if custom word features have
            wrong length.

        """
        assert (hasattr(self, 'duration') and hasattr(self, 'surprisal') and hasattr(self, 'wordonsets')), "Must load word features values and onset first!"
        if custom_wordfeats is not None:
            assert custom_wordfeats.shape[1] == len(self.wordonsets), "Please supply a list of word features that match the number of word onsets!"
        else:
            custom_wordfeats = [] # such that len() of it is defined and is 0

        # Are we adding word vectors:
        use_w2v = 'wordvectors' in features
        if use_w2v:
            features.remove('wordvectors')

        n_samples = int(np.ceil(srate * self.duration)) + 1 # +1 to account for 0th sample?
        nfeat = wordonset_feature + len(features) + use_w2v * self.vectordim + len(custom_wordfeats)
        feat = np.zeros((n_samples, nfeat))
        onset_samples = np.round(self.wordonsets * srate).astype(int)

        if wordonset_feature:
            feat[onset_samples, 0] = 1.

        for k, feat_val in enumerate(custom_wordfeats):
            assert len(feat_val) == len(self.wordonsets), "Feature #{:d} cannot be simply aligned with word onsets... ({:d} words mismatch)".format(k, abs(len(feat_val)-len(self.wordonsets)))
            if zscore_fun: # if not None
                if hasattr(zscore_fun, 'fit'): # sklearn object (which must have been fitted BEFOREHAND)
                    feat_val = zscore_fun.transform(feat_val)
                else:
                    feat_val = zscore_fun(feat_val)
            feat[onset_samples, k + int(wordonset_feature)] = feat_val

        current_idx = len(custom_wordfeats) + int(wordonset_feature)
        for k, feat_name in enumerate(features):
            assert len(getattr(self, feat_name)) == len(self.wordonsets), "Feat #{:d} cannot be simply aligned with onsets ({:d} words mismatch)".format(k, abs(len(feat_val)-len(self.wordonsets)))
            feat_val = getattr(self, feat_name)
            if zscore_fun: # if not None
                if hasattr(zscore_fun, 'fit'): # sklearn object (which must have been fitted BEFOREHAND)
                    feat_val = zscore_fun.transform(feat_val)
                else:
                    feat_val = zscore_fun(feat_val)
            feat[onset_samples, current_idx] = feat_val
            current_idx += 1

        return feat
