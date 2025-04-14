# -*- coding: utf-8 -*-
"""
Here are gathered all plotting functions.
Simply.

"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import colorsys
from matplotlib.patches import PathPatch
import numpy as np
from scipy import signal
from scipy.stats import ttest_rel
from .io import logging
from .utils import decorate_check_mne

logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__.split('.')[0])
try:
    import mne
except ImportError:
    # LOGGER.warning("MNE not installed, some functions will not work.")
    mne = None
    # User will be warned from specific function calls


PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb

def colormap_masked(ncolors=256, knee_index=None, cmap='inferno', alpha=0.3):
    """
    Create a colormap with value below a threshold being greyed out and transparent.
    
    Parameters
    ----------
    ncolors : int
        default to 256
    knee_index : int
        index from which transparency stops
        e.g. knee_index = np.argmin(abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))
    
    Returns
    -------
    cm : LinearSegmentedColormap
        Colormap instance
    """
    cm = plt.cm.get_cmap(cmap)(np.linspace(0, 1, ncolors))
    if knee_index is None:
        # Then map to pvals, as -log(p) between 0 and 3.5, and threshold at 0.05
        knee_index = np.argmin(abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))
    
    cm[:knee_index, :] = np.c_[cm[:knee_index, 0], cm[:knee_index, 1], cm[:knee_index, 2], alpha*np.ones((len(cm[:knee_index, 1])))]
    return LinearSegmentedColormap.from_list('my_colormap', cm)

def get_spatial_colors(info):
    "Create set of colours given info (i.e. channel locs) of raw mne object"
    loc3d = np.asarray([el['loc'][:3] for el in info['chs'] if (el['kind']==2 or el['kind']==1)])
    x, y, z = loc3d.T
    return _rgb(x, y, z)

@decorate_check_mne
def plot_interactive(data, info, ax=None, tmin=0., spatial_colors=True, picks=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    mne.viz.plot_evoked(mne.EvokedArray(data, info=info, tmin=tmin), axes=ax, spatial_colors=spatial_colors, picks=picks, show=False)
    # Remove N_ave: 
    for t in list(ax.texts): t.remove()

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

def plot_filterbank_output(signals, x=None, spacing=None, axis=-1):
    """
    Plot output coming out of a filterbank
    Each output of each channel is displayed on top of each other.
    """

    if spacing is None:
        spacing = signals.max()

    if x is None:
        x=range(signals.shape[axis])
        
    for k, filtered in enumerate(signals):
        plt.gca().set_prop_cycle(plt.cycler('color', COLORS[:signals.shape[2]]))
        if axis == -1:
            filtered = filtered.T
        plt.plot(x, filtered + k*spacing*2)

@decorate_check_mne
def topomap(arr, info, colorbar=True, ax=None, **kwargs):
    """
    Short-cut to mne topomap...

    Parameters
    ----------
    arr : ndarray (nchan,)
        Array of value to interpolate on a topographic map.
    info : mne.Info instance
        Contains EEG info (channel position for instance)
    
    Returns
    -------
    fig : Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    
    im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, **kwargs)
    if colorbar:
        plt.colorbar(im, ax=ax)
    return fig


@decorate_check_mne
def topoplot_array(data, pos, n_topos=1, titles=None):
    """
    Plotting topographic plot.
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
    for c in range(n_topos):
        inner_grid = outer_grid[c].subgridspec(1, 1)
        ax = plt.Subplot(fig, inner_grid[0])
        mne.viz.plot_topomap(data[:, c], pos, axes=ax, show=False)
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
    
def pairwise_boxplots(arr1, arr2, labels=['1', '2'], ax=None, linealpha=0.5,
                      boxalpha=0.5, add_signi=False):
    """
    Plot two boxplot of supplied series with small lines indicating pairwise 
    change between the two series.

    Parameters
    ----------
    arr1 : 1darray
        First series.
    arr2 : 1darray
        Second series.
    labels : TYPE, list[str]
        Xticklabels. The default is ['1', '2'].

    Returns
    -------
    Boxes, lines
    """
    assert len(arr1) == len(arr2), "Arrays must be of the same length"
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    
    props = {'widths': 0.4, 'patch_artist': True, 'showcaps': False, 'medianprops': {'color': 'k'}}
    b1 = ax.boxplot(arr1, positions=[0], **props)
    b2 = ax.boxplot(arr2, positions=[1], **props)
    for b in b1['boxes'] + b2['boxes']:
        b.set_facecolor(next(ax._get_lines.prop_cycler)['color'])
        b.set_alpha(boxalpha)
        
    lines = ax.plot(np.asarray([arr1[:], arr2[:]]), alpha=linealpha, zorder=10, marker='o', ms=5)
    
    if add_signi:
        pval = ttest_rel(arr1, arr2).pvalue
        significance_overlay(pval, [0, 1], ax=ax, barh=0.007, dh=0.01,)
    
    ax.set_xticklabels(labels)
    
    return {**b1, **b2}, lines

@decorate_check_mne
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
        info instance containing channel locations
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

def significance_overlay(pval, edges, height=None, color='k', yerr=None, dh=.05, barh=.05, fontsize=None, maxasterix=None, ax=None):
    """ 
    Annotate barplot (preferably, but any type really) with p-values.

    Parameters
    ----------
        pval: string to write or p-value for generating asterixes
        edges: data edge the bar
        height: height
        yerr: yerrs of all bars
        dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        barh: bar height in axes coordinates (0 to 1)
        fontsize: font size
        maxasterix: maximum number of asterixes to write (for very small p-values)
    Returns
    -------
        ax : plt.Axes
            Axis used.
    """
    if ax is None:
        ax = plt.gca()
    
    if height is None:
        height = ax.get_ylim()[1]

    if type(pval) is str:
        text = pval
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while pval < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = edges[0], height
    rx, ry = edges[1], height

    if yerr:
        ly += yerr[0]
        ry += yerr[1]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c=color, linewidth=1.8)

    kwargs_t = dict(ha='center', va='bottom')
    if fontsize is not None:
        kwargs_t['fontsize'] = fontsize

    ax.text(*mid, text, color=color, **kwargs_t)

    return ax

def prettify_boxplot(ax, desaturate=0.65):
    """
    Renders a prettier boxplot.
    
    The original boxplot ideally was drawn with ``seaborn``, or with :func:`plt.boxplot`
    while using ``patch_artist=True``.
    
    Parameters
    ----------
    ax : :class:`plt.AxesSubplot`
        Axis in which the boxplot is drawn.
    """
    patch = [p for p in ax.get_children() if isinstance(p, PathPatch)]
    lines = ax.get_lines()

    for p,l3 in zip(patch, np.reshape(lines, (len(patch), -1))):
        c = p.get_facecolor()
        hls = colorsys.rgb_to_hls(to_rgb(c))
        c = colorsys.hls_to_rgb(hls[0], desaturate*hls[1], hls[2])
        p.set_edgecolor(c)
        p.set_linewidth(1.5)
        for l in l3[:-1]: # last line is the median
            l.set_color(c)
            l.set_linewidth(1.5)
        l3[-1].set_linewidth(1.5)
        