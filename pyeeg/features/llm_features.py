"""
LLM-Based Feature Extraction

This module provides functionality for extracting word-level linguistic features
using language models, incorporating code from NNLM/lm_featurize.

Features:
- Surprisal: Information content of each word given its context (-log(P(word | context)), natural log)
- Entropy: Uncertainty in the token distribution (-sum(p * log(p)))
- KL Divergence: Difference between token distributions (KL(P||Q))
- Prediction Error: Surprisal normalized by entropy

The metric methods accept a ``return_shape`` argument controlling how the
returned vectors are aligned with the input tokens: ``"valid"`` (one value
per predicted token), ``"same"`` (a leading ``NaN`` is prepended so the
output length equals the input length), and ``"full"`` (one value per input
token; not supported by all metrics).

Based on NNLM/lm_featurize by Hugo Weissbart
"""

from dataclasses import dataclass

import numpy as np

from .._logging import LOGGER

try:
    import torch
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("torch is not installed. Instal torch to use this module (all optional deps installable via natmeeg[features])")


# Special tokens used in BPE tokenization
SEPARATOR = '\u0160'  # 'Ġ' - Byte pair encoding separator
NEWLINE = '\u010a'   # 'Ċ' - Newline token


@dataclass
class LLMFeatureConfig:
    """Configuration for LLM feature extraction.

    Attributes
    ----------
    model_name : str
        Name or path of the pretrained language model to load. If the name
        contains ``"gpt2"`` (case-insensitive), a :class:`GPT2LMHeadModel` is
        used; otherwise a generic :class:`AutoModel` is loaded. Defaults to
        ``"GroNLP/gpt2-small-dutch"``.
    device : str
        Device on which to run the model, e.g. ``"cpu"`` or ``"cuda"``.
        A CUDA device is only used if ``"cuda"`` is requested and CUDA is
        available. Defaults to ``"cpu"``.
    batch_size : int
        Batch size for feature extraction. Currently reserved; the extractor
        processes text with a single tokenizer call. Defaults to ``32``.
    max_length : int
        Maximum sequence length in tokens. Currently reserved; the extractor
        does not truncate inputs. Defaults to ``512``.
    cache_dir : Optional[str], optional
        Directory in which to cache the downloaded model and tokenizer.
        ``None`` uses the default Hugging Face cache. Defaults to ``None``.
    use_cache : bool
        Whether to enable Hugging Face model caching. Defaults to ``True``.
    return_shape : str, optional
        Default alignment of metric outputs (see the metric methods for
        per-call overrides). One of ``"valid"``, ``"same"``, or ``"full"``.
        Defaults to ``"valid"``.
    language : str, optional
        NLTK language name used by :meth:`bpe_to_words` to word-tokenize the
        decoded text when the fast (offset-mapping) alignment is unavailable
        (e.g. ``'nl'`` or ``'en'``). Defaults to ``'en'``.
    """
    model_name: str = "GroNLP/gpt2-small-dutch"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: str | None = None
    use_cache: bool = True
    return_shape: str = "valid"
    language: str = "en"


class LLMFeatureExtractor:
    """
    Extract word-level linguistic features using language models.

    This class loads a pretrained autoregressive language model and computes
    per-token metrics such as surprisal, entropy, KL divergence between
    adjacent token distributions, and prediction error. BPE-level features can
    be aggregated to word-level values via :meth:`reduce_features` after
    mapping tokens to words with :meth:`bpe_to_words`.

    Based on: NNLM/lm_featurize/metrics.py by Hugo Weissbart

    Parameters
    ----------
    config : LLMFeatureConfig, optional
        Configuration for the feature extractor. If ``None``, a default
        :class:`LLMFeatureConfig` is used.

    Attributes
    ----------
    config : LLMFeatureConfig
        Configuration of the extractor.
    _tokenizer : tokenizer or None
        Loaded Hugging Face tokenizer, populated by ``_initialize_model``.
    _model : model or None
        Loaded Hugging Face language model, populated by ``_initialize_model``.
    """

    def __init__(self, config: LLMFeatureConfig = None):
        """Initialize the extractor and load the language model.

        Parameters
        ----------
        config : LLMFeatureConfig, optional
            Configuration for the feature extractor. If ``None``, a default
            :class:`LLMFeatureConfig` is used.
        """
        if config is None:
            config = LLMFeatureConfig()
        self.config = config
        self._tokenizer = None
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the language model and tokenizer.

        Loads the tokenizer and model named by ``self.config.model_name``.
        GPT-2-style models are loaded as :class:`GPT2LMHeadModel` (providing
        ``logits`` outputs); other models are loaded with
        :class:`AutoModel`. Moves the model to ``config.device`` when CUDA is
        requested and available.

        Raises
        ------
        ImportError
            If the ``transformers`` library is not installed.
        """
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GPT2LMHeadModel,
            )
        except ImportError:
            LOGGER.error("transformers library not installed")
            raise

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # All models used here are causal (autoregressive) LMs and must expose
        # an LM head so that ``outputs.logits`` is available. ``AutoModel`` would
        # load the base encoder without a head and crash ``extract()`` for any
        # non-GPT-2 model name. GPT-2 is loaded explicitly for parity with the
        # original ``lm_featurize``; everything else uses ``AutoModelForCausalLM``.
        if "gpt2" in self.config.model_name.lower():
            self._model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        if self.config.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()

        LOGGER.info(f"Loaded model: {self.config.model_name}")

    def get_surprisal(self, logits: torch.Tensor, input_ids: torch.Tensor,
                    return_shape: str = None) -> torch.Tensor:
        """Calculate surprisal (-log(p)) for tokens.

        Surprisal is the negative natural logarithm of the probability the
        model assigns to each token given its preceding context, i.e.
        ``-log(P(token_i | tokens_<i))``. This is the information content of
        each token in nats.

        Based on: NNLM/lm_featurize/metrics.py

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape ``(seq_len, vocab_size)``. The first
            dimension is the token position (predictions at position ``i``
            are used to score token ``i+1``), and the second is the
            vocabulary dimension.
        input_ids : torch.Tensor
            Token IDs of shape ``(1, seq_len)`` (or ``(seq_len,)``) whose
            values index into the vocabulary dimension of ``logits``.
        return_shape : str, optional
            Alignment of the returned vector with the input tokens:

            - ``"valid"``: one surprisal value per predicted token, i.e.
              ``seq_len - 1`` values (token 0 has no preceding context and is
              not scored).
            - ``"same"``: ``seq_len`` values; a leading ``NaN`` is prepended
              so the output aligns positionally with ``input_ids``.
            - ``"full"``: ``seq_len`` values, scoring each token (including
              the first) with its own logit row.

            If ``None`` (default), ``self.config.return_shape`` is used.

        Returns
        -------
        surprisal : torch.Tensor
            1D tensor of surprisal values in nats, with length depending on
            ``return_shape`` (see above).

        Raises
        ------
        ValueError
            If ``return_shape`` is not one of ``"valid"``, ``"same"``, or
            ``"full"``.
        """
        if return_shape is None:
            return_shape = self.config.return_shape

        # Compute surprisal directly from log-softmax (log P) rather than
        # taking ``log`` of a softmax probability. Working in log-space avoids
        # catastrophic loss of precision for low-probability tokens, which
        # matter for surprisal (rare/surprising tokens have high surprisal).
        logp = F.log_softmax(logits, dim=1)
        ids = input_ids.flatten()

        if return_shape == 'valid':
            # Surprisal of tokens 1..N-1 predicted by positions 0..N-2.
            return -logp[:-1, :].gather(1, ids[1:].unsqueeze(1)).squeeze(1)
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, -logp[:-1, :].gather(1, ids[1:].unsqueeze(1)).squeeze(1)))
        elif return_shape == 'full':
            return -logp[:, :].gather(1, ids.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")

    def get_entropy(self, logits: torch.Tensor, return_shape: str = None) -> torch.Tensor:
        """Calculate entropy of token distributions.

        Computes the Shannon entropy in nats of the predictive distribution
        at each token position: ``-sum(p * log(p))`` over the vocabulary.

        Based on: NNLM/lm_featurize/metrics.py

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape ``(seq_len, vocab_size)``.
        return_shape : str, optional
            Alignment of the returned vector with the input tokens:

            - ``"valid"``: ``seq_len - 1`` values (the entropy of the last
              logits row is dropped, matching the surprisal alignment).
            - ``"same"``: ``seq_len`` values; a leading ``NaN`` is prepended
              so the output aligns positionally with the input tokens.
            - ``"full"``: ``seq_len`` values, one per logits row.

            If ``None`` (default), ``self.config.return_shape`` is used.

        Returns
        -------
        entropy : torch.Tensor
            1D tensor of entropy values in nats, with length depending on
            ``return_shape`` (see above).

        Raises
        ------
        ValueError
            If ``return_shape`` is not one of ``"valid"``, ``"same"``, or
            ``"full"``.
        """
        if return_shape is None:
            return_shape = self.config.return_shape

        if return_shape == 'valid':
            return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)[:-1]
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)[:-1]))
        elif return_shape == 'full':
            return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), 1)
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")

    def get_kl_divergence(self, logits: torch.Tensor, return_shape: str = None) -> torch.Tensor:
        """Compute KL divergence between current and previous state.

        Measures how much the predictive distribution at each token position
        differs from the distribution at the immediately preceding position,
        i.e. ``KL(P_t || P_{t-1})`` computed from the log-softmax rows of
        ``logits``.

        Based on: NNLM/lm_featurize/metrics.py

        Parameters
        ----------
        logits : torch.Tensor
            Raw model logits of shape ``(seq_len, vocab_size)``.
        return_shape : str, optional
            Alignment of the returned vector with the input tokens:

            - ``"valid"``: ``seq_len - 1`` values, one KL divergence per
              adjacent pair of rows.
            - ``"same"``: ``seq_len`` values; a leading ``NaN`` is prepended
              so the output aligns positionally with the input tokens.
            - ``"full"``: **not supported** by this metric; requesting it
              raises :class:`ValueError`.

            If ``None`` (default), ``self.config.return_shape`` is used.

        Returns
        -------
        kl : torch.Tensor
            1D tensor of KL divergence values (nats), with length depending on
            ``return_shape`` (see above). The first token has no previous
            state and is assigned ``NaN`` in ``"same"`` mode.

        Raises
        ------
        ValueError
            If ``return_shape`` is not one of ``"valid"``, ``"same"``, or
            ``"full"``. Note that ``"full"`` is rejected for this metric
            (only ``"valid"`` and ``"same"`` are implemented).
        """
        if return_shape is None:
            return_shape = self.config.return_shape

        P = F.log_softmax(logits, dim=1)

        if return_shape == 'valid':
            # KL between each adjacent pair of distributions: KL(P_t || P_{t-1})
            # for t = 1..N-1, i.e. N-1 values.
            return F.kl_div(P[:-1, :], P[1:, :], reduction='none', log_target=True).sum(1)
        elif return_shape == 'same':
            nan_val = torch.ones(1, device=logits.device) * float('nan')
            return torch.cat((nan_val, F.kl_div(P[:-1, :], P[1:, :], reduction='none', log_target=True).sum(1)))
        else:
            raise ValueError(f"Unknown return_shape: {return_shape}")

    def get_prediction_error(self, surprisal: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """Calculate prediction error (surprisal/entropy).

        Prediction error is the ratio of surprisal to entropy, normalizing the
        surprisal of each token by how uncertain the model was at that
        position. Zero entropy values are replaced with ``1e-10`` before the
        division to avoid division by zero.

        Parameters
        ----------
        surprisal : torch.Tensor
            1D tensor of surprisal values (nats) for each token.
        entropy : torch.Tensor
            1D tensor of entropy values (nats) for each token, same length as
            ``surprisal``.

        Returns
        -------
        prediction_error : torch.Tensor
            1D tensor of ``surprisal / entropy`` values, same length as the
            inputs.
        """
        entropy_safe = entropy.clone()
        entropy_safe[entropy_safe == 0] = 1e-10
        return surprisal / entropy_safe

    def get_tokens(self, input_ids: torch.Tensor) -> list[str]:
        """Convert input IDs to token strings.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(1, seq_len)`` (or ``(seq_len,)``).

        Returns
        -------
        tokens : list of str
            The decoded token strings, in sequence order. Tokens may be BPE
            subword units (e.g. prefixed with the ``Ġ`` separator).
        """
        return self._tokenizer.convert_ids_to_tokens(input_ids.flatten())

    def bpe_to_words(self, tokens: list[str], sep: str = SEPARATOR, newline: str = NEWLINE,
                     lang: str = 'nl') -> list[list[int]]:
        """Map BPE tokens to word indices.

        Joins the BPE tokens into a string (replacing the BPE separator with
        a space and the newline token with a newline), tokenizes it into words
        with NLTK, then greedily matches each word back to the contiguous run
        of BPE tokens that compose it.

        Based on: NNLM/lm_featurize/utils.py

        Parameters
        ----------
        tokens : list of str
            BPE token strings as returned by :meth:`get_tokens`.
        sep : str, optional
            BPE separator substring marking a word boundary. Defaults to
            :data:`SEPARATOR` (``'Ġ'``).
        newline : str, optional
            Substring used for the newline token. Defaults to
            :data:`NEWLINE` (``'Ċ'``).
        lang : str, optional
            NLTK language name passed to :func:`nltk.tokenize.word_tokenize`
            (e.g. ``'nl'`` or ``'en'``). Defaults to ``'nl'``.

        Returns
        -------
        indices_map : list of list of int
            For each word, the list of BPE token indices composing that word.

        Raises
        ------
        ImportError
            If the ``nltk`` library is not installed.
        RecursionError
            If a word cannot be matched to BPE tokens within 100 tokens.
        """
        try:
            from nltk.tokenize import word_tokenize
        except ImportError:
            LOGGER.error("NLTK not installed")
            raise

        tokens_clean = word_tokenize(''.join(tokens).replace(sep, ' ').replace(newline, '\n'), language=lang)

        indices_map = []
        k = 0

        for t in tokens_clean:
            bytepairs = []
            test_token = tokens[k].strip(sep).strip(newline)
            if test_token != '':
                bytepairs.append(k)

            while test_token != t:
                k += 1
                if k == len(tokens): break
                test_token += tokens[k].strip(sep).strip(newline)
                if test_token != '':
                    bytepairs.append(k)
                if len(bytepairs) > 100:
                    raise RecursionError(f"Can't match token: {t}")

            indices_map.append(bytepairs)
            k += 1
            if k >= len(tokens): break

        return indices_map

    def bpe_to_words_fast(self, text: str, tokens: list[str]) -> list[list[int]]:
        """Map BPE tokens to word indices via the tokenizer's offset mapping.

        Faster and more robust alternative to :meth:`bpe_to_words` that uses
        the tokenizer's character offset mapping (available only for
        "fast" Rust tokenizers) to group BPE tokens by the orthographic word
        they belong to. Tokens that carry no span (special tokens) are skipped.

        Parameters
        ----------
        text : str
            The original input text that was tokenized.
        tokens : list of str
            BPE token strings as returned by :meth:`get_tokens` (used only as
            a fallback / sanity reference when offset mapping is unavailable).

        Returns
        -------
        indices_map : list of list of int
            For each word, the list of BPE token indices composing that word.

        Raises
        ------
        NotImplementedError
            If the loaded tokenizer is not a "fast" tokenizer and does not
            support ``return_offsets_mapping``.
        """
        if not getattr(self._tokenizer, "is_fast", False):
            raise NotImplementedError(
                "bpe_to_words_fast requires a fast tokenizer with offset "
                "mapping support; use bpe_to_words() instead."
            )
        enc = self._tokenizer(text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]

        # Build character-level word boundaries from the original text.
        # Iterate over the offset spans and group consecutive tokens that
        # belong to the same whitespace-delimited word.
        indices_map = []
        current_group = []
        last_word_end = -1

        for idx, (start, end) in enumerate(offsets):
            if start == end:
                # Special token with empty span; skip.
                if current_group:
                    indices_map.append(current_group)
                    current_group = []
                last_word_end = end
                continue
            # A new word starts when there is a gap (whitespace) since the
            # previous token's end.
            if current_group and start > last_word_end:
                indices_map.append(current_group)
                current_group = []
            current_group.append(idx)
            last_word_end = end

        if current_group:
            indices_map.append(current_group)

        return indices_map

    def reduce_features(self, features: dict[str, np.ndarray], indices_map: list[list[int]]) -> dict[str, np.ndarray]:
        """Reduce BPE-level features to word-level.

        Aggregates per-token feature arrays into one value per word using the
        BPE-to-word index mapping produced by :meth:`bpe_to_words`. The
        aggregation rule depends on the feature name:

        - ``tokens``: subword strings are concatenated and BPE markers
          (separator/newline) removed.
        - ``entropy``: the value of the **last** BPE token of the word.
        - ``kl_divergence``: the sum of the first and last BPE token values
          (or just the first when the word is a single token).
        - all other features (e.g. ``surprisal``): the **sum** over the BPE
          tokens of the word.

        If both ``surprisal`` and ``entropy`` are present, a ``prediction_error``
        feature is appended as ``surprisal / entropy`` (zero entropy replaced
        with ``1e-10``).

        Based on: NNLM/lm_featurize/utils.py

        Parameters
        ----------
        features : dict of str -> np.ndarray
            BPE-level features. Must contain a ``tokens`` entry (array of
            token strings) plus one or more numeric feature arrays, all of
            length equal to the number of BPE tokens.
        indices_map : list of list of int
            For each word, the list of BPE token indices, as produced by
            :meth:`bpe_to_words`.

        Returns
        -------
        reduced_features : dict of str -> np.ndarray
            Word-level features, one value per word. A ``prediction_error``
            entry is added when both ``surprisal`` and ``entropy`` are
            present in the input.
        """
        from functools import reduce as functools_reduce

        def sum_reduce(x):
            return functools_reduce(lambda x, y: x+y, x)

        reduced_features = {}

        for feat_name, feat_array in features.items():
            if feat_name == 'tokens':
                reduced_tokens = []
                for bpe_indices in indices_map:
                    token_str = ''.join([features['tokens'][i] for i in bpe_indices]).replace(sep, '').replace(newline, '')
                    reduced_tokens.append(token_str)
                reduced_features[feat_name] = np.array(reduced_tokens)
            elif feat_name == 'entropy':
                reduced_values = []
                for bpe_indices in indices_map:
                    reduced_values.append(feat_array[bpe_indices[-1]])
                reduced_features[feat_name] = np.array(reduced_values)
            elif feat_name == 'kl_divergence':
                reduced_values = []
                for bpe_indices in indices_map:
                    if len(bpe_indices) > 1:
                        reduced_values.append(feat_array[bpe_indices[0]] + feat_array[bpe_indices[-1]])
                    else:
                        reduced_values.append(feat_array[bpe_indices[0]])
                reduced_features[feat_name] = np.array(reduced_values)
            else:
                reduced_values = []
                for bpe_indices in indices_map:
                    reduced_values.append(sum_reduce(feat_array[bpe_indices]))
                reduced_features[feat_name] = np.array(reduced_values)

        if 'surprisal' in reduced_features and 'entropy' in reduced_features:
            entropy_safe = reduced_features['entropy'].copy()
            entropy_safe[entropy_safe == 0] = 1e-10
            reduced_features['prediction_error'] = reduced_features['surprisal'] / entropy_safe

        return reduced_features

    def extract(self, text: str, features: list[str] = None,
                return_word_level: bool = True) -> dict[str, np.ndarray]:
        """Extract features from text.

        Runs the language model over the tokenized text and returns the
        requested metrics. Metrics are always computed with
        ``return_shape='same'`` (leading ``NaN`` for the first token), and are
        optionally aggregated from BPE tokens to word level.

        Parameters
        ----------
        text : str
            Input text to process.
        features : list of str, optional
            Names of the features to compute. Valid entries are
            ``'surprisal'``, ``'entropy'``, ``'kl_divergence'``,
            ``'prediction_error'``, and ``'tokens'``. Requesting
            ``'prediction_error'`` also computes ``surprisal`` and ``entropy``
            internally. If ``None`` (default), all four metric features are
            computed.
        return_word_level : bool, optional
            If ``True`` (default), BPE-level features are aggregated to
            word-level using :meth:`bpe_to_words` and :meth:`reduce_features`
            (with ``lang='en'``). If ``False``, raw per-token arrays are
            returned.

        Returns
        -------
        result : dict of str -> np.ndarray
            Feature name to values array mapping. When
            ``return_word_level=True``, each array has one entry per word; the
            ``tokens`` entry (if requested) contains the reconstructed word
            strings. When ``return_word_level=False``, each feature array has
            one entry per BPE token (with a leading ``NaN`` where applicable)
            and ``tokens`` contains the decoded token strings. Features are
            only present if they were requested (or required by
            ``prediction_error``); if nothing was requested, the dict is
            empty.
        """
        if features is None:
            features = ['surprisal', 'entropy', 'kl_divergence', 'prediction_error']

        inputs = self._tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids']

        if self.config.device == "cuda" and torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        result = {}

        if 'surprisal' in features or 'prediction_error' in features:
            result['surprisal'] = self.get_surprisal(logits, input_ids, 'same').detach().cpu().numpy()

        if 'entropy' in features or 'prediction_error' in features:
            result['entropy'] = self.get_entropy(logits, 'same').detach().cpu().numpy()

        if 'kl_divergence' in features:
            result['kl_divergence'] = self.get_kl_divergence(logits, 'same').detach().cpu().numpy()

        if 'tokens' in features:
            result['tokens'] = np.array(self.get_tokens(input_ids))

        if return_word_level and len(result) > 0:
            tokens = self.get_tokens(input_ids)
            # Use the fast offset-mapping alignment when the tokenizer supports
            # it; fall back to the string-matching alignment otherwise.
            if getattr(self._tokenizer, "is_fast", False):
                indices_map = self.bpe_to_words_fast(text, tokens)
            else:
                indices_map = self.bpe_to_words(tokens, lang=self.config.language)
            result = self.reduce_features(result, indices_map)

        return result
