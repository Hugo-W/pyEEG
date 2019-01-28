#! python
# -*- coding: utf-8 -*-
"""
Input/Output
~~~~~~~~~~~~~~~

Desctription
==============

Input/output of data from files:
     - import EEG data from .set files or raw files
     - import features (word onsets, envelopes, etc.)

Module to correctly extract Matlab files,
especially EEGLAB .set files no matter if the version is >=7.3 and
take the data to MNE format.

*Author: Hugo Weissbart*

"""
import numpy as np
import pandas as pd
import h5py # for mat file version >= 7.3
import mne
from mne.preprocessing.ica import ICA
#import os
#os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
from scipy.io import loadmat

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
            for k, v in fid.iteritems():
                data[k] = v.value
    return data


def _loadEEGLAB_data(fname):
    """"
    Load eeglab structure and returns data, channel names,
    sampling frequency and other useful data...
    """
    with h5py.File(fname, mode='r') as fid:
        eeg = fid['EEG/data'].value
        srate = fid['EEG/srate'].value

        time = fid['EEG/times'].value

        events = np.zeros(fid['EEG/event/latency'].shape)
        event_type = []
        for k in range(len(events)):
            events[k] = fid[fid['EEG/event/latency'][k][0]].value
            event_type.append(''.join([chr(i) for i in fid[fid['EEG/event/type'][k][0]] ]))

        chnames = []
        for k in range(fid['EEG/chanlocs/labels'].shape[0]):
            chnames.append(''.join([chr(i) for i in fid[fid['EEG/chanlocs/labels'][k][0]] ]))

        return eeg, srate, time, events, event_type, chnames

def load_ica_matrices(fname):
    """
    Load ICA matrices from EEGLAB structure .set file
    """
    try:
        eeg = loadmat(fname)['EEG']
        weights = eeg[0, 0][19]
        winv = eeg[0, 0][17]
        sphere = eeg[0, 0][18]

    except NotImplementedError:
        print("WARNING: must use hdf5 extraction of .set file info")
        with h5py.File(fname, mode='r') as fid:
            weights = fid['EEG/icaweights'].value
            winv = fid['EEG/icawinv'].value
            sphere = fid['EEG/icasphere'].value

    return weights, winv, sphere

def eeglab2mne(fname, montage='standard_1020', event_id=None, load_ica=False):
    """
    Get an EEGLAB dataset into a MNE Raw object.

    Parameters
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
            if load_ica True

    Notes
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
        eeg, srate, _, events, etype, ch_names = _loadEEGLAB_data(fname)
        info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types='eeg', montage=montage_mne)
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

def get_exact_duration(fname):
    "Extract value of speech segment duration from Praat file..."
    with open(fname, 'r') as f:
        headers = f.readlines(80)
        duration = float(headers[4])
    return duration

def get_word_onsets(filepath):
    "Simply load word lists and respective onsets for a given sound file/story parts..."
    csv = pd.read_csv(filepath)
    return csv.word.get_values(), csv.onset.get_values()

def get_wordfreq(filepath):
    "Simply load word lists and respective onsets for a given sound file/story parts..."
    csv = pd.read_csv(filepath)
    wf = csv.frequency.get_values()
    # Replace unknown by (global, form my stories) median value:
    wf[wf==-1] = 111773390
    # Transform into log proba, normalized by maximum count, roughly equals to total count:
    wf = -np.log(wf/3.2137e12)
    return wf

def get_surprisal(filepath, eps=1e-12):
    "Simply load word lists and respective onsets for a given sound file/story parts..."
    df = pd.read_csv(filepath, delim_whitespace=True, usecols=[0, 1, 2, 3, 4])
    surprisal = df.loc[np.logical_and(df.Word != '</s>', df.Word != '\''), 'P(NET)'].get_values()
    return -np.log(surprisal + eps)

def align_word_features(word_onsets, word_feat_list, speech_duration, sr, wordonset_feature=True, zscore_fun=None):
    """Check that we have the correct number of elements for each features
    Will output a time series with spike for each word features.
    For that reason, we need the duration and sampling rate as inputs.

    Parameters
    ----------
    word_onsers : array-like
        List of word onset in seconds.
    word_feat_list : List or array (nfeat, nwords)
        List of word level features arrays.
    speech_duration : float
        Duration of speech utterance for this list of words.
    sr : float
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

    Returns
    -------
    feat : array (ntimes, nfeat)
        Array of spikes, with the word feature value at word onsets.
        ntimes ~ speech_duration * sr

    """
    Nsamples = int(np.ceil(sr * speech_duration)) + 1 # +1 to account for 0th sample?
    nfeat = len(word_feat_list) + int(wordonset_feature)
    feat = np.zeros((Nsamples, nfeat))
    t_spikes = np.zeros((Nsamples,))
    onset_samples = np.round(word_onsets * sr).astype(int)
    t_spikes[onset_samples] = 1.

    if wordonset_feature:
        feat[:, 0] = t_spikes

    for k, feat_val in enumerate(word_feat_list):
        assert len(feat_val)==len(word_onsets), "Feature #{:d} cannot be simply aligned with word onsets... ({:d} words mismatch)".format(k, abs(len(feat_val)-len(word_onsets)))
        if zscore_fun: # if not None
            if hasattr(zscore_fun, 'fit'): # sklearn object (which must have been fitted BEFOREHAND)
                feat_val = zscore_fun.transform(feat_val)
            else:
                feat_val = zscore_fun(feat_val)
        feat[onset_samples, k + int(wordonset_feature)] = feat_val

    return feat
