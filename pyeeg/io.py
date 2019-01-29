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


def _loadEEGLAB_data(fname):
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
        eeg, srateate, _, _, _, ch_names = _loadEEGLAB_data(fname)
        info = mne.create_info(ch_names=ch_names, sfreq=srateate, ch_types='eeg', montage=montage_mne)
        raw = mne.io.RawArray(eeg.T, info)

    if load_ica:
        weights, winv, sphere = load_ica_matrices(fname)
        ica = ICA(n_components=winv.shape[1], max_pca_components=winv.shape[1])
        ica.fit(raw, decim=2, start=1., stop=60.)
        ica.unmixing_matrix_ = weights.dot(sphere.dot(ica.pca_components_.T))
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        return raw, ica
    else:
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

    Attributes
    ----------

    Examples
    --------
    >>> from pyeeg.io import WordLevelFeatures
    >>> path_features = '/media/hw2512/../word_onsets.txt'
    >>> wordfeats = WordLevelFeatures(path_features)
    >>> wordfeats.load_onsets(path_onsets)
    >>> wordfeats.load_surprisal(path_surprisal)
    >>> wordfeats.df.head()
    index   | onsets    | surprisal
    0       | 1.12      | 3
    ...

    Note
    ----
    For now, the class assumes that the data are stored in a given format that depended
    on the processing done for the work on surprisal. However it ought to be extended to
    be bale to generate surprisal/word frequency/word onsets/word vectors for a given audio
    files along with its transcript.

    TODO
    ----
    Extend the class to create word-feature of choices on-the-fly

    """

    def __init__(self, path_praat_env=None, path_surprisal=None, path_wordvectors=None,
                 path_wordfrequency=None, path_wordonsets=None, path_transcript=None,
                 rnnlm_model=None, path_audio=None):

        if path_praat_env is not None:
            self.duration = extract_duration_praat(path_praat_env)
        self.surprisal = load_surprisal_values(path_surprisal)
        self.wordfrequency = load_wordfreq_values(path_wordfrequency)
        self.wordonsets, self.wordlist = get_word_onsets(path_wordonsets)
        self.audio = path_audio
        self.rnnlm = rnnlm_model
        self.transcript = None
        if path_transcript is not None:
            with open(path_transcript, 'r') as fid:
                raw = fid.read()
            self.transcript = raw

        self.wordvectors = None
        self.vectordim = 0



    def align_word_features(self, srate, wordonset_feature=True, surprisal=True,
                            wordfreq=True, wordvect=False, zscore_fun=None, custom_wordfeats=None):
        """Check that we have the correct number of elements for each features
        Will output a time series with spike for each word features.
        For that reason, we need the duration and sampling rate as inputs.

        Parameters
        ----------
        srate : float
            Sampling rate at which to compute spike-like features
        wordonset_feature : boolean
            Whether to use word onset (always "on" feature) in the
            feature array (Default: True).
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
        feat : array (ntimes, nfeat)
            Array of spikes, with the word feature value at word onsets.
            ntimes ~ speech_duration * srate

        Raises
        ------
        AssertionError
            If the duration of speech part has not been extracted or if custom word features have
            wrong length.

        TODO
        ----
        Add entropy and word vectors!!

        """
        assert (hasattr(self, 'duration') and hasattr(self, 'surprisal') and hasattr(self, 'wordonsets')), "Must load word features values and onset first!"
        if custom_wordfeats is not None:
            assert custom_wordfeats.shape[1] == len(self.wordonsets), "Please supply a list of word features that match the number of word onsets!"
        else:
            custom_wordfeats = [] # such that len() of it is defined and is 0

        n_samples = int(np.ceil(srate * self.duration)) + 1 # +1 to account for 0th sample?
        nfeat = wordonset_feature + surprisal + wordfreq + wordvect * self.vectordim + len(custom_wordfeats)
        feat = np.zeros((n_samples, nfeat))
        t_spikes = np.zeros((n_samples,))
        onset_samples = np.round(self.wordonsets * srate).astype(int)
        t_spikes[onset_samples] = 1.

        if wordonset_feature:
            feat[:, 0] = t_spikes

        for k, feat_val in enumerate(custom_wordfeats):
            assert len(feat_val) == len(self.wordonsets), "Feature #{:d} cannot be simply aligned with word onsets... ({:d} words mismatch)".format(k, abs(len(feat_val)-len(self.wordonsets)))
            if zscore_fun: # if not None
                if hasattr(zscore_fun, 'fit'): # sklearn object (which must have been fitted BEFOREHAND)
                    feat_val = zscore_fun.transform(feat_val)
                else:
                    feat_val = zscore_fun(feat_val)
            feat[onset_samples, k + int(wordonset_feature)] = feat_val


        feat_to_use = []
        if wordfreq:
            feat_to_use.append(self.wordfrequency)
        if surprisal:
            feat_to_use.append(self.surprisal)

        current_idx = len(custom_wordfeats) + int(wordonset_feature)
        for feat_val in feat_to_use:
            assert len(feat_val) == len(self.wordonsets), "Feature #{:d} cannot be simply aligned with word onsets... ({:d} words mismatch)".format(k, abs(len(feat_val)-len(self.wordonsets)))
            if zscore_fun: # if not None
                if hasattr(zscore_fun, 'fit'): # sklearn object (which must have been fitted BEFOREHAND)
                    feat_val = zscore_fun.transform(feat_val)
                else:
                    feat_val = zscore_fun(feat_val)
            feat[onset_samples, current_idx] = feat_val
            current_idx += 1

        return feat
