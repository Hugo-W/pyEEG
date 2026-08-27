"""
Feature Extraction Pipeline

This module provides a pipeline for composing multiple feature extractors
and aligning their outputs with neural signals. :class:`FeaturePipeline`
orchestrates LLM and syntactic feature extractors defined by
:class:`FeatureSpec` entries, optionally aligning the word-level features to
a neural signal sampled at a given rate via :class:`AlignmentHandler`.
:class:`StimulusEncoder` is a high-level convenience wrapper around the
pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .._logging import LOGGER

try:
    from .llm_features import LLMFeatureExtractor, LLMFeatureConfig
    LLMEXTRACTOR_AVAILABLE = True
except ImportError:
    LLMEXTRACTOR_AVAILABLE = False
from .syntactic_features import SyntacticFeatureExtractor, ParserConfig
from .alignment import AlignmentHandler, TextGrid

try:
    from .acoustic import AcousticFeatureConfig, AcousticFeatureExtractor
    ACOUSTICEXTRACTOR_AVAILABLE = True
except ImportError:
    ACOUSTICEXTRACTOR_AVAILABLE = False


@dataclass
class FeatureSpec:
    """Specification for a feature to be extracted.

    Attributes
    ----------
    name : str
        Unique name of the extractor within the pipeline. Also used as a
        prefix for the keys of the extracted features
        (``"{name}_{feature}"``).
    extractor_type : str
        Type of the extractor to instantiate. One of ``"llm"``
        (:class:`~pyeeg.features.llm_features.LLMFeatureExtractor`) or
        ``"syntactic"``
        (:class:`~pyeeg.features.syntactic_features.SyntacticFeatureExtractor`).
    features : list of str
        Names of the individual features to extract from this extractor. For
        LLM extractors these are e.g. ``"surprisal"``, ``"entropy"``,
        ``"kl_divergence"``, ``"prediction_error"``, or ``"tokens"``; for
        syntactic extractors e.g. ``"depth"``, ``"opening"``, ``"closing"``,
        or ``"tree_height"``. Defaults to an empty list (extractor defaults
        apply).
    config : dict or None, optional
        Optional per-extractor configuration. For ``"llm"``: ``model_name``
        and ``device``; for ``"syntactic"``: ``parser_name`` and
        ``language``. Defaults to ``None``.
    """

    name: str
    extractor_type: str
    features: List[str] = field(default_factory=list)
    config: Optional[Dict] = None


@dataclass
class PipelineConfig:
    """Configuration for the feature extraction pipeline.

    Attributes
    ----------
    feature_specs : list of FeatureSpec
        Ordered list of :class:`FeatureSpec` entries describing the
        extractors to run. Defaults to an empty list.
    alignment_config : dict or None, optional
        Optional alignment configuration. Currently supports a single key,
        ``sampling_rate`` (float, default ``1000.0``), used to construct the
        :class:`~pyeeg.features.alignment.AlignmentHandler`. If ``None``, no
        alignment handler is created and features are returned unaligned.
        Defaults to ``None``.
    normalization : str
        Normalization method applied to the extracted features by
        :meth:`FeaturePipeline.normalize_features` when used from
        :class:`StimulusEncoder`. One of ``"none"``, ``"zscore"``, or
        ``"minmax"``. Defaults to ``"none"``.
    cache_features : bool
        Whether to cache extracted features between calls. **Currently
        reserved**: no caching is implemented in the pipeline. Defaults to
        ``True``.
    """

    feature_specs: List[FeatureSpec] = field(default_factory=list)
    alignment_config: Optional[Dict] = None
    normalization: str = "none"
    cache_features: bool = True


class UserDefinedExtractor:
    """Wraps a user-provided callable as a feature extractor.

    The callable should accept the stimulus (text or audio signal) and
    return a dict mapping feature names to numpy arrays.
    """

    def __init__(self, func, name="custom"):
        self._func = func
        self.name = name

    def extract(self, stimulus, **kwargs):
        return self._func(stimulus, **kwargs)


class FeaturePipeline:
    """Pipeline for extracting and aligning multiple features from stimuli.

    Runs the extractors declared in the configuration's ``feature_specs``
    over an input text, then either returns the word-level feature arrays
    directly (by word position) or aligns them to a neural signal using the
    configured :class:`~pyeeg.features.alignment.AlignmentHandler` and an
    optional Praat :class:`~pyeeg.features.alignment.TextGrid`.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration listing the feature specs, alignment options,
        and normalization method.

    Attributes
    ----------
    config : PipelineConfig
        Pipeline configuration.
    _extractors : dict of str -> object
        Mapping of extractor name (from ``FeatureSpec.name``) to the
        instantiated extractor object.
    _alignment_handler : AlignmentHandler or None
        Alignment handler constructed from ``config.alignment_config``, or
        ``None`` if no alignment was configured.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline and its extractors and alignment handler.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration listing the feature specs, alignment
            options, and normalization method.
        """
        self.config = config
        self._extractors: Dict[str, object] = {}
        self._alignment_handler: Optional[AlignmentHandler] = None
        self._initialize_extractors()
        self._initialize_alignment()
    
    def _initialize_extractors(self):
        """Initialize all feature extractors.

        Iterates over ``self.config.feature_specs`` and instantiates the
        appropriate extractor for each spec: an
        :class:`~pyeeg.features.llm_features.LLMFeatureExtractor` for
        ``extractor_type='llm'`` (only if the LLM backend is importable) or a
        :class:`~pyeeg.features.syntactic_features.SyntacticFeatureExtractor`
        for ``extractor_type='syntactic'``. Extractors are stored under
        ``spec.name`` in ``self._extractors``; unknown extractor types are
        skipped.
        """
        for spec in self.config.feature_specs:
            cfg = spec.config or {}
            if spec.extractor_type == 'llm':
                llm_config = LLMFeatureConfig(
                    model_name=cfg.get('model_name', 'GroNLP/gpt2-small-dutch'),
                    device=cfg.get('device', 'cpu')
                )
                if LLMEXTRACTOR_AVAILABLE:
                    self._extractors[spec.name] = LLMFeatureExtractor(llm_config)
            elif spec.extractor_type == 'syntactic':
                parser_config = ParserConfig(
                    parser_name=cfg.get('parser_name', 'stanford'),
                    language=cfg.get('language', 'en')
                )
                self._extractors[spec.name] = SyntacticFeatureExtractor(parser_config)
            elif spec.extractor_type == 'custom':
                func = cfg.get('func')
                if func is not None:
                    self._extractors[spec.name] = UserDefinedExtractor(func, spec.name)
            elif spec.extractor_type == 'acoustic':
                if ACOUSTICEXTRACTOR_AVAILABLE:
                    acoustic_config = AcousticFeatureConfig(
                        sampling_rate=cfg.get('sampling_rate', 16000),
                        features=cfg.get('features') or ['envelope'],
                        envelope_method=cfg.get('envelope_method', 'hilbert'),
                    )
                    self._extractors[spec.name] = AcousticFeatureExtractor(acoustic_config)
    
    def _initialize_alignment(self):
        """Initialize alignment handler.

        If ``self.config.alignment_config`` is set, constructs an
        :class:`~pyeeg.features.alignment.AlignmentHandler` using the
        ``sampling_rate`` key of the alignment config (default ``1000.0``)
        and stores it in ``self._alignment_handler``. Otherwise the handler
        is left as ``None``.
        """
        if self.config.alignment_config:
            sampling_rate = self.config.alignment_config.get('sampling_rate', 1000.0)
            self._alignment_handler = AlignmentHandler(sampling_rate)
    
    def extract(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract features from text and optionally align to signal.

        Runs every configured extractor over ``text`` and collects the
        resulting features. If no alignment is available (no ``textgrid`` and
        no configured alignment handler), the features are returned as
        word-position arrays; otherwise they are aligned to signal samples
        using the word intervals of the ``textgrid``.

        When syntactic extractors return per-word dicts, their entries are
        converted to word-position arrays of length ``len(text.split())``.

        Parameters
        ----------
        text : str
            Input text to process.
        textgrid : TextGrid, optional
            Optional Praat :class:`~pyeeg.features.alignment.TextGrid`
            providing word intervals for aligning features to the signal. If
            ``None`` and no alignment handler is configured, features are
            returned unaligned. If ``None`` but an alignment handler exists, a
            warning is logged and an empty :class:`TextGrid` is used (yielding
            an empty aligned array).
        signal_length : int, optional
            Length of the neural signal in samples. If ``None``, it is derived
            from the TextGrid end time and the alignment sampling rate.

        Returns
        -------
        result : dict of str -> ndarray
            Feature name to array mapping. Keys are ``"{extractor_name}_{feature}"``.
            When unaligned, each array has one entry per word (or per token for
            LLM token-level features). When aligned, each array has one entry
            per signal sample.
        metadata : dict
            Metadata describing the extraction: ``text`` (input text),
            ``text_length`` (character count), ``feature_specs`` (configured
            spec names), ``aligned`` (whether the features were aligned to a
            signal), and, when aligned, ``n_samples`` (number of aligned
            samples) and ``sampling_rate`` (sampling rate used).
        """
        metadata = {
            'text': text,
            'text_length': len(text),
            'feature_specs': [spec.name for spec in self.config.feature_specs],
            'aligned': textgrid is not None
        }
        
        all_features: Dict[str, Dict] = {}
        
        for spec in self.config.feature_specs:
            extractor = self._extractors.get(spec.name)
            if extractor is None:
                continue
            
            if spec.extractor_type == 'llm':
                features = extractor.extract(text, spec.features)
            elif spec.extractor_type == 'syntactic':
                features = extractor.extract_to_dict(text, spec.features)
            elif spec.extractor_type == 'custom':
                features = extractor.extract(text)
            else:
                continue
            
            if features:
                all_features[spec.name] = features
        
        if textgrid is None and self._alignment_handler is None:
            result = {}
            for extractor_name, feat_dict in all_features.items():
                for feat_name, values in feat_dict.items():
                    if isinstance(values, dict):
                        words = text.split()
                        arr = np.zeros(len(words))
                        for pos, val in values.items():
                            if pos < len(words):
                                arr[pos] = val
                        result[f"{extractor_name}_{feat_name}"] = arr
                    elif isinstance(values, np.ndarray):
                        result[f"{extractor_name}_{feat_name}"] = values
            metadata['aligned'] = False
            return result, metadata
        
        if textgrid is None:
            LOGGER.warning("No TextGrid provided and no alignment handler configured")
            textgrid = TextGrid()
        
        result = {}
        aligned = np.array([])
        for extractor_name, feat_dict in all_features.items():
            # ``align_word_features`` expects a mapping of
            # ``{word_index: {feature_name: value}}``.  Convert from the two
            # conventions used by the extractors:
            #   LLM:       {feature_name: ndarray}  (one value per word)
            #   Syntactic: {feature_name: {word_index: value}}
            # Non-numeric features (e.g. "tokens" strings) and NaN values are
            # skipped so they do not corrupt the float alignment matrix.
            n_words = len(text.split())
            word_map: Dict[int, Dict[str, float]] = {}
            for feat_name, values in feat_dict.items():
                if isinstance(values, np.ndarray):
                    if values.dtype.kind in ('f', 'i', 'u'):
                        for i in range(min(len(values), n_words)):
                            val = float(values[i])
                            if not np.isnan(val):
                                word_map.setdefault(i, {})[feat_name] = val
                elif isinstance(values, dict):
                    for pos, val in values.items():
                        try:
                            pos_int = int(pos)
                            fval = float(val)
                        except (ValueError, TypeError):
                            continue
                        if not np.isnan(fval):
                            word_map.setdefault(pos_int, {})[feat_name] = fval
            aligned, feat_names = self._alignment_handler.align_word_features(
                word_map, textgrid, signal_length
            )
            for i, feat_name in enumerate(feat_names):
                result[f"{extractor_name}_{feat_name}"] = aligned[:, i]
        
        metadata['aligned'] = True
        metadata['n_samples'] = aligned.shape[0] if aligned.size > 0 else 0
        metadata['sampling_rate'] = self._alignment_handler.sampling_rate
        
        return result, metadata

    def extract_audio(
        self,
        signal: np.ndarray,
        srate: float,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract acoustic features from an audio signal.

        Runs every configured acoustic extractor over ``signal`` and collects
        the resulting features, prefixed with each extractor's name.

        Parameters
        ----------
        signal : ndarray
            Audio signal as a 1D numpy array.
        srate : float
            Sampling rate of the audio signal in Hz.
        signal_length : int, optional
            Reserved for compatibility with the text path; not used by
            acoustic extraction.

        Returns
        -------
        result : dict of str -> ndarray
            Feature name to array mapping. Keys are
            ``"{extractor_name}_{feature}"``.
        metadata : dict
            Metadata describing the extraction: ``stimulus_type`` (``"audio"``),
            ``sampling_rate`` (provided sampling rate), ``feature_specs``
            (configured acoustic spec names), and ``n_samples`` (length of the
            input signal).
        """
        metadata = {
            'stimulus_type': 'audio',
            'sampling_rate': srate,
            'feature_specs': [
                spec.name for spec in self.config.feature_specs
                if spec.extractor_type == 'acoustic'
            ],
            'n_samples': len(signal) if signal is not None else 0,
        }

        all_features: Dict[str, Dict] = {}

        for spec in self.config.feature_specs:
            if spec.extractor_type != 'acoustic':
                continue
            extractor = self._extractors.get(spec.name)
            if extractor is None:
                continue
            features = extractor.extract(signal, srate)
            if features:
                all_features[spec.name] = features

        result = {}
        for extractor_name, feat_dict in all_features.items():
            for feat_name, values in feat_dict.items():
                result[f"{extractor_name}_{feat_name}"] = values

        return result, metadata

    def normalize_features(
        self,
        features: Dict[str, np.ndarray],
        method: str = "zscore"
    ) -> Dict[str, np.ndarray]:
        """Normalize features.

        Applies the requested normalization independently to each feature
        array in ``features``.

        Parameters
        ----------
        features : dict of str -> ndarray
            Feature name to values array mapping.
        method : str, optional
            Normalization method. One of:

            - ``"none"``: return the input unchanged.
            - ``"zscore"``: subtract the mean and divide by the standard
              deviation (``(x - mean) / std``). If the standard deviation is
              zero, only the mean is subtracted.
            - ``"minmax"``: scale to the ``[0, 1]`` range via
              ``(x - min) / (max - min)``. If all values are equal, the
              minimum is subtracted.

            Any other value leaves the array unchanged (same behavior as
            ``"none"``). Defaults to ``"zscore"``.

        Returns
        -------
        normalized : dict of str -> ndarray
            Feature name to normalized values array mapping, with the same
            keys as the input.
        """
        if method == 'none':
            return features
        
        result = {}
        for name, arr in features.items():
            if method == 'zscore':
                mean = np.nanmean(arr)
                std = np.nanstd(arr)
                if std > 0:
                    result[name] = (arr - mean) / std
                else:
                    result[name] = arr - mean
            elif method == 'minmax':
                min_val = np.nanmin(arr)
                max_val = np.nanmax(arr)
                if max_val > min_val:
                    result[name] = (arr - min_val) / (max_val - min_val)
                else:
                    result[name] = arr - min_val
            else:
                result[name] = arr
        return result


class StimulusEncoder:
    """High-level interface for stimulus feature extraction.

    Wraps a :class:`FeaturePipeline` and provides convenience methods for
    adding LLM or syntactic feature extractors, configuring alignment, and
    encoding text into feature arrays (optionally normalized according to the
    pipeline configuration).

    Parameters
    ----------
    pipeline_config : PipelineConfig, optional
        Configuration for the underlying pipeline. If ``None``, a default
        :class:`PipelineConfig` (no feature specs, no alignment) is used.

    Attributes
    ----------
    pipeline : FeaturePipeline
        The underlying feature extraction pipeline.
    """

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        """Initialize the encoder with an optional pipeline configuration.

        Parameters
        ----------
        pipeline_config : PipelineConfig, optional
            Configuration for the underlying pipeline. If ``None``, a default
            :class:`PipelineConfig` is used.
        """
        if pipeline_config is None:
            pipeline_config = PipelineConfig()
        self.pipeline = FeaturePipeline(pipeline_config)
    
    def add_llm_features(
        self,
        features: List[str] = None,
        model_name: str = "GroNLP/gpt2-small-dutch",
        name: str = "llm"
    ):
        """Add LLM-based features to the pipeline.

        Appends an LLM :class:`FeatureSpec` to the pipeline configuration and
        re-initializes the extractors so the new spec takes effect.

        Parameters
        ----------
        features : list of str, optional
            LLM features to extract, e.g. ``"surprisal"``, ``"entropy"``,
            ``"kl_divergence"``, ``"prediction_error"``, or ``"tokens"``. If
            ``None`` (default), ``["surprisal", "entropy", "kl_divergence"]``
            is used.
        model_name : str, optional
            Hugging Face model name or path to load. Defaults to
            ``"GroNLP/gpt2-small-dutch"``.
        name : str, optional
            Extractor name used as prefix for the output feature keys and as
            the pipeline spec name. Defaults to ``"llm"``.

        Returns
        -------
        None
        """
        if features is None:
            features = ['surprisal', 'entropy', 'kl_divergence']
        spec = FeatureSpec(
            name=name,
            extractor_type='llm',
            features=features,
            config={'model_name': model_name}
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()
    
    def add_syntactic_features(
        self,
        features: List[str] = None,
        parser_name: str = "stanford",
        name: str = "syntactic"
    ):
        """Add syntactic features to the pipeline.

        Appends a syntactic :class:`FeatureSpec` to the pipeline configuration
        and re-initializes the extractors so the new spec takes effect.

        Parameters
        ----------
        features : list of str, optional
            Syntactic features to extract, e.g. ``"depth"``, ``"opening"``,
            ``"closing"``, or ``"tree_height"``. If ``None`` (default),
            ``["depth", "opening", "closing"]`` is used.
        parser_name : str, optional
            Parser to use (e.g. ``"stanford"``). Defaults to ``"stanford"``.
        name : str, optional
            Extractor name used as prefix for the output feature keys and as
            the pipeline spec name. Defaults to ``"syntactic"``.

        Returns
        -------
        None
        """
        if features is None:
            features = ['depth', 'opening', 'closing']
        spec = FeatureSpec(
            name=name,
            extractor_type='syntactic',
            features=features,
            config={'parser_name': parser_name}
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()

    def add_custom_extractor(self, func, name='custom', features=[]):
        """Add a user-defined extractor to the pipeline.

        Appends a ``"custom"`` :class:`FeatureSpec` wrapping ``func`` to the
        pipeline configuration and re-initializes the extractors so the new
        spec takes effect. The callable is invoked with the stimulus (text or
        audio signal) and should return a dict mapping feature names to numpy
        arrays.

        Parameters
        ----------
        func : callable
            User-provided extractor callable, called as ``func(stimulus)``.
            Must return a dict of ``{feature_name: numpy array}``.
        name : str, optional
            Extractor name used as prefix for the output feature keys and as
            the pipeline spec name. Defaults to ``"custom"``.
        features : list of str, optional
            Optional feature names for the spec. Defaults to an empty list.

        Returns
        -------
        None
        """
        spec = FeatureSpec(
            name=name,
            extractor_type='custom',
            features=features,
            config={'func': func}
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()

    def add_acoustic_features(
        self,
        features=None,
        sampling_rate=16000,
        name='acoustic',
        **kwargs
    ):
        """Add acoustic features to the pipeline.

        Appends an ``"acoustic"`` :class:`FeatureSpec` to the pipeline
        configuration and re-initializes the extractors so the new spec takes
        effect.

        Parameters
        ----------
        features : list of str, optional
            Acoustic features to extract (e.g. ``"envelope"``). If ``None``
            (default), ``["envelope"]`` is used.
        sampling_rate : float, optional
            Sampling rate of the audio signal in Hz. Defaults to ``16000``.
        name : str, optional
            Extractor name used as prefix for the output feature keys and as
            the pipeline spec name. Defaults to ``"acoustic"``.
        **kwargs
            Additional acoustic configuration options passed through to the
            extractor's config (e.g. ``envelope_method``).

        Returns
        -------
        None
        """
        if features is None:
            features = ['envelope']
        config = {'sampling_rate': sampling_rate, 'features': features}
        config.update(kwargs)
        spec = FeatureSpec(
            name=name,
            extractor_type='acoustic',
            features=features,
            config=config
        )
        self.pipeline.config.feature_specs.append(spec)
        self.pipeline._initialize_extractors()

    def set_alignment(self, sampling_rate: float = 1000.0):
        """Set alignment configuration.

        Configures the pipeline to align extracted features to a neural signal
        sampled at ``sampling_rate`` and re-initializes the alignment handler.

        Parameters
        ----------
        sampling_rate : float, optional
            Sampling rate of the neural signal in Hz. Defaults to ``1000.0``.

        Returns
        -------
        None
        """
        self.pipeline.config.alignment_config = {
            'sampling_rate': sampling_rate
        }
        self.pipeline._initialize_alignment()
    
    def encode(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract and align features from text.

        Runs the pipeline's :meth:`FeaturePipeline.extract`, then applies the
        normalization configured in the pipeline configuration (unless it is
        ``"none"``).

        Parameters
        ----------
        text : str
            Input text to process.
        textgrid : TextGrid, optional
            Optional Praat :class:`~pyeeg.features.alignment.TextGrid` for
            aligning features to the signal.
        signal_length : int, optional
            Length of the neural signal in samples.

        Returns
        -------
        features : dict of str -> ndarray
            Feature name to values array mapping (see
            :meth:`FeaturePipeline.extract` for the shape semantics). If
            normalization is configured, arrays are normalized.
        metadata : dict
            Metadata describing the extraction (see
            :meth:`FeaturePipeline.extract`).
        """
        features, metadata = self.pipeline.extract(
            text, textgrid, signal_length
        )
        if self.pipeline.config.normalization != 'none':
            features = self.pipeline.normalize_features(
                features,
                self.pipeline.config.normalization
            )
        return features, metadata

    def encode_audio(
        self,
        signal: np.ndarray,
        srate: float,
        signal_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Extract acoustic features from an audio signal.

        Runs the pipeline's :meth:`FeaturePipeline.extract_audio`, then applies
        the normalization configured in the pipeline configuration (unless it is
        ``"none"``).

        Parameters
        ----------
        signal : ndarray
            Audio signal as a 1D numpy array.
        srate : float
            Sampling rate of the audio signal in Hz.
        signal_length : int, optional
            Reserved for compatibility with the text path; not used by
            acoustic extraction.

        Returns
        -------
        features : dict of str -> ndarray
            Feature name to values array mapping (see
            :meth:`FeaturePipeline.extract_audio`). If normalization is
            configured, arrays are normalized.
        metadata : dict
            Metadata describing the extraction (see
            :meth:`FeaturePipeline.extract_audio`).
        """
        features, metadata = self.pipeline.extract_audio(
            signal, srate, signal_length
        )
        if self.pipeline.config.normalization != 'none':
            features = self.pipeline.normalize_features(
                features,
                self.pipeline.config.normalization
            )
        return features, metadata

    def encode_to_array(
        self,
        text: str,
        textgrid: Optional[TextGrid] = None,
        signal_length: Optional[int] = None
    ) -> np.ndarray:
        """Extract features and return as a single array.

        Runs :meth:`encode` and stacks the resulting feature arrays
        column-wise into a single 2D array. Feature columns are ordered by
        sorted feature name.

        Parameters
        ----------
        text : str
            Input text to process.
        textgrid : TextGrid, optional
            Optional Praat :class:`~pyeeg.features.alignment.TextGrid` for
            aligning features to the signal.
        signal_length : int, optional
            Length of the neural signal in samples.

        Returns
        -------
        array : ndarray
            2D array of shape ``(n_samples, n_features)`` with one column per
            extracted feature (ordered by sorted feature name), or an empty
            array if no features were extracted.
        """
        features, metadata = self.encode(text, textgrid, signal_length)
        if not features:
            return np.array([])
        feature_names = sorted(features.keys())
        stacked = np.column_stack([features[name] for name in feature_names])
        return stacked