# -*- coding: utf-8 -*-
"""
Here are gathered all plotting functions.
Simply.

"""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from scipy import signal
import mne

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb

def get_spatial_colors(info):
    "Create set of colours given info (i.e. channel locs) of raw mne object"
    loc3d = np.asarray([el['loc'][:3] for el in info['chs']])
    x, y, z = loc3d.T
    return _rgb(x, y, z)

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
    Plotting topographic plot.
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
    for c in range(n_topos):
        inner_grid = outer_grid[c].subgridspec(1, 1)
        ax = plt.Subplot(fig, inner_grid[0])
        im, _ = mne.viz.plot_topomap(data[:, c], pos, axes=ax, show=False)
        ax.set(title=titles[c])
        fig.add_subplot(ax)

def plot_trf_signi(trf, reject, time_highlight=None, spatial_colors=True, info=None, ax=None, shades=None, **kwargs):
    "Plot trf with significant portions highlighted and with thicker lines"
    trf.plot(ax=ax, **kwargs)

    if spatial_colors:
        assert info is not None, "To use spatial colouring, you must supply raw.info instance"
        colors = get_spatial_colors(info)

    signi_trf = np.ones_like(reject) * np.nan
    list_axes = ax if ax is not None else plt.gcf().axes
    for feat, cax in enumerate(list_axes):
        if shades is None:
            color_shade = 'w' if np.mean(to_rgb(plt.rcParams['axes.facecolor'])) < .5 else [.2, .2, .2]
        else:
            color_shade = shades
        if time_highlight is None:
            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                            where=np.any(reject[:, feat, :], 1),
                            color=color_shade, alpha=0.2)
        else: # fill regions of time of interest
            toi = np.zeros_like(trf.times, dtype=bool)
            for tlims in time_highlight[feat]:
                toi = np.logical_or(toi, np.logical_and(trf.times >= tlims[0], trf.times < tlims[1]))

            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                            where=toi,
                            color=color_shade, alpha=0.2)
        lines = cax.get_lines()
        for k, l in enumerate(lines):
            if spatial_colors:
                l.set_color(np.r_[colors[k], 0.3])
            signi_trf[reject[:, feat, k], feat, k] = l.get_data()[1][reject[:, feat, k]]
        newlines = cax.plot(trf.times, signi_trf[:, feat, :], linewidth=4)
        if spatial_colors:
            for k, l in enumerate(newlines):
                l.set_color(colors[k])
    if ax is None:
        return plt.gcf()

def plots_topogrid(x, y, info, yerr=None, mask=None):
    """
    Display a series of plot arranged in a topographical grid.
    Shaded error bars can be displayed, as well as masking for
    significance portions of data.

    Parameters
    ----------
    x : 1d-array
        Absciss
    y : ndarray
        Data, (ntimes, nchans)
    info : mne.info instance
        info instance ocntaining channel locations
    yerr : ndarry
        Error for shaded areas
    mask : ndarray <bool>
        Boolean array to highlight significant portions of data
        Same shape as y

    Returns
    -------
    fig : figure
    """
    fig = plt.figure(figsize=(12, 10))
    for ax, chan_idx in mne.viz.topo.iter_topography(info,
                                                     fig_facecolor=(36/256, 36/256, 36/256, 0), axis_facecolor='#333333',
                                                     axis_spinecolor='white', fig=fig):
        ax.plot(x, y[:, chan_idx])
        if yerr is not None:
            ax.fill_between(x, y[:, chan_idx] - yerr[:, chan_idx], y[:, chan_idx] + yerr[:, chan_idx],
                            facecolor='C0', edgecolor='C0', linewidth=0, alpha=0.5)
        if mask is not None:
            ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=mask[:, chan_idx].T,
                            facecolor='C2', edgecolor='C2', linewidth=0, alpha=0.5)
        ax.hlines(0, xmin=x[0], xmax=x[-1], linestyle='--', alpha=0.5)
        # Change axes spine color if contains significant portion
        if mask is not None:
            if any(mask[:, chan_idx]):
                for _, v in ax.spines.items():
                    v.set_color('C2')
    return fig
