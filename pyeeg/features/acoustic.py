"""
Acoustic feature extraction for audio stimuli.

This module provides :class:`AcousticFeatureExtractor`, a thin wrapper that
exposes the low-level audio feature helpers of :mod:`pyeeg.utils`,
:mod:`pyeeg.preprocess` and :mod:`pyeeg.ratemap` behind a single class
compatible with the conventions of the feature extraction pipeline
(:mod:`pyeeg.features.pipeline`).
"""

from dataclasses import dataclass, field

import numpy as np

from .._logging import LOGGER
from ..preprocess import apply_filterbank, create_filterbank
from ..ratemap import make_rate_map
from ..utils import cochleogram, signal_envelope


def _compression_to_string(compression):
    """Map a numeric compression factor (or name) to the string expected by
    :func:`pyeeg.ratemap.make_rate_map`.

    The compiled rate-map extension only accepts the strings ``'cuberoot'``,
    ``'log'`` and ``'none'``. Numeric compression exponents are mapped as
    follows: ``1/3`` (the default) -> ``'cuberoot'``, ``0``/``None`` ->
    ``'none'``, anything else -> ``'none'``.
    """
    if isinstance(compression, str):
        return compression
    if compression is None or compression == 0:
        return 'none'
    if np.isclose(compression, 1 / 3):
        return 'cuberoot'
    return 'none'


@dataclass
class AcousticFeatureConfig:
    """Configuration for :class:`AcousticFeatureExtractor`.

    Attributes
    ----------
    sampling_rate : float
        Default sampling rate (Hz) of the audio signal, used when ``srate`` is
        not passed to :meth:`AcousticFeatureExtractor.extract`. Defaults to
        ``16000.0``.
    features : list of str
        Names of the features to extract. Any subset of ``'envelope'``,
        ``'filterbank'`` and ``'gammatone'``. Defaults to ``['envelope']``.
    envelope_method : str
        Envelope method passed to :func:`pyeeg.utils.signal_envelope` (one of
        ``'hilbert'``, ``'rectify'``, ``'subs'``). Defaults to ``'hilbert'``.
    envelope_cutoff : float
        Low-pass cutoff (Hz) of the envelope. Defaults to ``20.0``.
    envelope_comp_factor : float
        Compression factor of the envelope (``env**comp_factor``). Defaults to
        ``1/3``.
    filterbank_freqs : list of float
        Boundary frequencies (Hz) of the filterbank bands, passed to
        :func:`pyeeg.preprocess.create_filterbank`. Defaults to
        ``[100, 500, 1000, 2000, 4000, 8000]``.
    filterbank_kwargs : dict
        Additional keyword arguments for the filter design function used by
        :func:`pyeeg.preprocess.create_filterbank` (e.g. ``N`` and ``rs`` for
        Chebyshev type II filters). Defaults to ``{'N': 2, 'rs': 3}``, matching
        the usage in :mod:`pyeeg.cca`.
    gammatone_nchannels : int
        Number of frequency channels of the gammatone rate map. Defaults to
        ``32``.
    gammatone_lowcf : float
        Centre frequency of the lowest channel (Hz). Defaults to ``80.0``.
    gammatone_highcf : float
        Centre frequency of the highest channel (Hz). Defaults to ``8000.0``.
    gammatone_frameshift : float
        Interval between successive frames (ms). Defaults to ``8.0``.
    gammatone_compression : float or str
        Compression of the rate map. Either a numeric compression exponent
        (``1/3`` is mapped to ``'cuberoot'``) or one of the strings
        ``'cuberoot'``, ``'log'``, ``'none'`` accepted by
        :func:`pyeeg.ratemap.make_rate_map`. Defaults to ``1/3``.
    """

    sampling_rate: float = 16000.0
    features: list = field(default_factory=lambda: ['envelope'])
    # Envelope params
    envelope_method: str = 'hilbert'
    envelope_cutoff: float = 20.0
    envelope_comp_factor: float = 1 / 3
    # Filterbank params
    filterbank_freqs: list = field(
        default_factory=lambda: [100, 500, 1000, 2000, 4000, 8000]
    )
    filterbank_kwargs: dict = field(default_factory=lambda: {'N': 2, 'rs': 3})
    # Gammatone/ratemap params
    gammatone_nchannels: int = 32
    gammatone_lowcf: float = 80.0
    gammatone_highcf: float = 8000.0
    gammatone_frameshift: float = 8.0  # ms
    gammatone_compression: float = 1 / 3


class AcousticFeatureExtractor:
    """Extract acoustic features from an audio signal.

    Parameters
    ----------
    config : AcousticFeatureConfig, optional
        Configuration controlling which features to extract and the parameters
        of each underlying function. If ``None``, a default
        :class:`AcousticFeatureConfig` is used.

    Attributes
    ----------
    config : AcousticFeatureConfig
        Extractor configuration.
    """

    def __init__(self, config: AcousticFeatureConfig | None = None):
        """Initialize the extractor with an optional configuration.

        Parameters
        ----------
        config : AcousticFeatureConfig, optional
            Extractor configuration. If ``None``, a default configuration is
            used.
        """
        if config is None:
            config = AcousticFeatureConfig()
        self.config = config

    def extract(
        self, signal: np.ndarray, srate: float | None = None
    ) -> dict[str, np.ndarray]:
        """Extract acoustic features from an audio signal.

        Parameters
        ----------
        signal : ndarray (nsamples,) or (nsamples, nchannels)
            Audio signal.
        srate : float, optional
            Sampling rate. If None, uses config.sampling_rate.

        Returns
        -------
        features : dict
            Mapping from feature name to ndarray.
            - 'envelope': broadband envelope (nsamples_env,) or
              (nsamples_env, nchannels) for multi-channel input.
            - 'filterbank': filtered signal per band (nfilters, nsamples) or
              (nfilters, nsamples, nchannels) for multi-channel input.
            - 'gammatone': rate map / cochleogram (nchannels, nframes) or
              (nchannels, nframes, nchans) for multi-channel input.
        """
        if srate is None:
            srate = self.config.sampling_rate
        signal = np.asarray(signal)
        if signal.ndim not in (1, 2):
            raise ValueError(
                f"signal must be 1D or 2D, got {signal.ndim} dimensions"
            )

        features = {}
        for name in self.config.features:
            if name == 'envelope':
                features[name] = self._extract_envelope(signal, srate)
            elif name == 'filterbank':
                features[name] = self._extract_filterbank(signal, srate)
            elif name == 'gammatone':
                features[name] = self._extract_gammatone(signal, srate)
            else:
                LOGGER.warning("Unknown acoustic feature %r, skipping", name)
        return features

    def _extract_envelope(self, signal: np.ndarray, srate: float) -> np.ndarray:
        """Broadband envelope via :func:`pyeeg.utils.signal_envelope`.

        For multi-channel input, the envelope is computed per channel and
        stacked column-wise, yielding ``(nsamples_env, nchannels)``.
        """
        if signal.ndim == 1:
            return signal_envelope(
                signal,
                srate,
                cutoff=self.config.envelope_cutoff,
                method=self.config.envelope_method,
                comp_factor=self.config.envelope_comp_factor,
            )
        cols = [
            signal_envelope(
                signal[:, i],
                srate,
                cutoff=self.config.envelope_cutoff,
                method=self.config.envelope_method,
                comp_factor=self.config.envelope_comp_factor,
            )
            for i in range(signal.shape[1])
        ]
        return np.column_stack(cols)

    def _extract_filterbank(self, signal: np.ndarray, srate: float) -> np.ndarray:
        """Band-pass filtered signal via :func:`pyeeg.preprocess.create_filterbank`
        and :func:`pyeeg.preprocess.apply_filterbank`.

        ``apply_filterbank`` natively handles 2D ``(nsamples, nchannels)``
        input, so multi-channel data is filtered without looping.

        Boundary frequencies at or above the Nyquist frequency (``srate / 2``)
        cannot be realized by digital filter design, so they are skipped with a
        warning (e.g. the default ``8000`` Hz boundary at a 16 kHz sampling
        rate).
        """
        nyquist = srate / 2.0
        freqs = [
            f for f in self.config.filterbank_freqs if f < nyquist
        ]
        skipped = len(self.config.filterbank_freqs) - len(freqs)
        if skipped:
            LOGGER.warning(
                "Skipping %d filterbank boundary(ies) at/above Nyquist "
                "(srate/2 = %.1f Hz)", skipped, nyquist
            )
        if not freqs:
            raise ValueError(
                "No valid filterbank boundaries below the Nyquist frequency "
                f"(srate/2 = {nyquist:.1f} Hz)"
            )
        fbank = create_filterbank(freqs, srate, **self.config.filterbank_kwargs)
        return apply_filterbank(signal, fbank)

    def _extract_gammatone(self, signal: np.ndarray, srate: float) -> np.ndarray:
        """Gammatone rate map via :func:`pyeeg.ratemap.make_rate_map`.

        Falls back to :func:`pyeeg.utils.cochleogram` when the compiled
        rate-map extension is unavailable (or fails). For multi-channel input,
        the rate map is computed per channel and stacked along a new trailing
        axis, yielding ``(nchannels, nframes, nchans)``.
        """
        if signal.ndim == 1:
            return self._gammatone_1d(signal, srate)
        maps = [self._gammatone_1d(signal[:, i], srate) for i in range(signal.shape[1])]
        return np.stack(maps, axis=-1)

    def _gammatone_1d(self, signal: np.ndarray, srate: float) -> np.ndarray:
        """Compute the gammatone rate map of a 1D signal."""
        c = self.config
        try:
            return make_rate_map(
                signal,
                int(srate),
                c.gammatone_lowcf,
                c.gammatone_highcf,
                c.gammatone_nchannels,
                c.gammatone_frameshift,
                0,
                _compression_to_string(c.gammatone_compression),
            )
        except Exception:  # noqa: BLE001 - fall back on any extension failure
            LOGGER.warning(
                "make_rate_map failed, falling back to cochleogram", exc_info=True
            )
            return cochleogram(
                signal,
                srate,
                shift=c.gammatone_frameshift,
                nchannels=c.gammatone_nchannels,
                fmin=c.gammatone_lowcf,
                fmax=c.gammatone_highcf,
                comp_factor=_compression_to_string(c.gammatone_compression),
            )