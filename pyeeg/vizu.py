# -*- coding: utf-8 -*-
"""
Here are gathered all plotting functions.
Simply.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import mne

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def plot_filterbank(fbank):
    """
    Plotting a filterbank as created by :func:`pyeeg.preprocess.create_filterbank`
    """
    #plt.plot(w/np.pi,20*np.log10(abs(H)+1e-6))
    signal.freqz(np.stack(fbank)[:, 0, :].T[..., np.newaxis], np.stack(fbank)[:, 1, :].T[..., np.newaxis],
                 plot=lambda w, h: plt.plot(w/np.pi, np.abs(h.T)))
    plt.title('Filter Magnitude Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude (not in dB)')

def plot_filterbank_output(signals, spacing=None, axis=-1):
    """
    Plot output coming out of a filterbank
    Each output of each channel is displayed on top of each other.
    """

    if spacing is None:
        spacing = signals.max()

    for k, filtered in enumerate(signals):
        plt.gca().set_prop_cycle(plt.cycler('color', COLORS[:signals.shape[2]]))
        if axis == -1:
            filtered = filtered.T
        plt.plot(filtered + k*spacing*2)
        
def topoplot_array(data, pos, n_topos=1, titles=None):
    """
    Plotting topographic plot from spatial filters.
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
    for c in range(n_topos):
        inner_grid = outer_grid[c].subgridspec(1, 1)
        ax = plt.Subplot(fig, inner_grid[0])
        im, _ = mne.viz.plot_topomap(data[:,c], pos, axes=ax, show=False)
        ax.set(title="CC #{:d}".format(c+1))
        fig.add_subplot(ax)