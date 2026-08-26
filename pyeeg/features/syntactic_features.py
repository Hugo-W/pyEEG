"""
Syntactic Feature Extraction

This module provides functionality for extracting syntactic features from
constituency trees, incorporating code from process_txt/pyProcess/parseMetrics.py.

Features:
- Depth: Depth of each node in the parse tree
- Opening nodes: Number of branches opening at each leaf
- Closing nodes: Number of branches closing at each leaf
- Tree height: Height of the whole parse tree, repeated per word

The :class:`SyntacticFeatureExtractor` class wraps constituency parsing
(Stanford Parser for English or Alpino for Dutch) and computes per-word
feature vectors that can be aligned with neural signals.

Based on parseMetrics.py by Hugo Weissbart
"""

import os
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .._logging import LOGGER

try:
    from nltk.tree import Tree
    from nltk.parse.stanford import StanfordParser
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    Tree = None
    NLTK_AVAILABLE = False

try:
    import alpinonaf
    ALPINO_AVAILABLE = True
except ImportError:
    ALPINO_AVAILABLE = False


@dataclass
class ParserConfig:
    """Configuration for the external constituency parser.

    Parameters
    ----------
    parser_name : str
        Name of the parser to use: ``"stanford"`` or ``"alpino"``.
        Default: ``"stanford"``.
    parser_path : str, optional
        Path to the parser binary or jar file (e.g. ``stanford-parser.jar``).
        Used by :meth:`SyntacticFeatureExtractor.get_stanford_tree` when no
        explicit path is given. Default: ``None``.
    model_path : str, optional
        Path to the parser model file. Currently unused by the extraction
        methods, which rely on the ``STANFORD_MODELS`` environment variable.
        Default: ``None``.
    language : str
        Language of the input text, passed to ``nltk.sent_tokenize``
        (supported: ``"en"``, ``"nl"``, ``"fr"``, or any language code
        understood by NLTK). Default: ``"en"``.
    timeout : int
        Timeout in seconds for parser calls. Reserved for parser backends
        that support timeouts; not currently enforced by the extraction
        methods. Default: 30.
    """
    parser_name: str = "stanford"
    parser_path: Optional[str] = None
    model_path: Optional[str] = None
    language: str = "en"
    timeout: int = 30


class SyntacticFeatureExtractor:
    """
    Extract syntactic features from text using constituency parsing.

    Wraps an external constituency parser (Stanford Parser for English or
    Alpino for Dutch) and computes per-word syntactic features (depth,
    opening, closing, tree height) from the resulting parse trees. Feature
    values are returned in word order, so they can be aligned with neural
    signals via :mod:`pyeeg.features.alignment`.

    Based on: process_txt/pyProcess/parseMetrics.py by Hugo Weissbart
    """
    
    def __init__(self, config: ParserConfig = None):
        """Initialize the extractor with a parser configuration.

        Parameters
        ----------
        config : ParserConfig, optional
            Configuration for the external parser. If ``None``, a default
            :class:`ParserConfig` is used.
        """
        if config is None:
            config = ParserConfig()
        self.config = config
    
    def get_stanford_tree(self, sentences: List[str], path_to_jar: Optional[str] = None) -> List[Tree]:
        """Parse sentences using the Stanford Parser.

        Loads the Stanford Parser via NLTK and parses each sentence,
        returning one parse tree per sentence. The parser classpath is
        derived from ``path_to_jar`` (or ``config.parser_path``); if the
        path does not exist, an attempt is made to locate
        ``stanford-parser.jar`` with the ``locate`` command. The
        ``STANDFORD_MODELS`` environment variable must be set, pointing to
        the Stanford models directory.

        Based on parseMetrics.py.

        Parameters
        ----------
        sentences : list of str
            Sentences to parse. Each element is parsed independently.
        path_to_jar : str, optional
            Path to ``stanford-parser.jar``. If ``None``,
            ``config.parser_path`` is used. Default: ``None``.

        Returns
        -------
        trees : list of nltk.tree.Tree
            One parse tree per sentence, in input order.

        Raises
        ------
        ImportError
            If NLTK is not available.
        IOError
            If ``stanford-parser.jar`` cannot be found or the
            ``STANFORD_MODELS`` environment variable is not set.
        """
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK required for Stanford parsing")
        
        LOGGER.info("Loading Stanford Parser...")
        
        if path_to_jar is None:
            path_to_jar = self.config.parser_path
        
        if path_to_jar is not None:
            path_to_jar = os.path.expanduser(path_to_jar)
            if not os.path.exists(path_to_jar):
                import subprocess
                try:
                    path_to_jar = subprocess.check_output(['bash', '-c', 'locate stanford-parser.jar']).strip('\n').decode('utf-8')
                except Exception:
                    raise IOError("stanford-parser.jar not found")
        
        if path_to_jar:
            os.environ['CLASSPATH'] = path_to_jar
        
        if os.getenv('STANFORD_MODELS') is None:
            raise IOError("STANFORD_MODELS not set")
        
        parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        LOGGER.info("Stanford Parser loaded")
        
        parse = parser.raw_parse_sents(sentences)
        trees = []
        for treelist in parse:
            for tree in treelist:
                trees.append(tree)
        return trees
    
    def get_alpinopy_tree(self, sentences: List[str]) -> List[Tree]:
        """Parse Dutch sentences using the Alpino parser.

        Uses the ``alpinonaf`` wrapper to parse each sentence. Based on
        parseMetrics.py.

        Notes
        -----
        This method has filesystem side effects: it sets
        ``tempfile.tempdir`` to ``/tmp/alpino`` (creating that directory if
        missing), writes each sentence to a temporary file with a
        ``alpino-`` prefix, and relies on the parser writing its output to a
        file named ``penn_output.txt`` in the current working directory,
        which is read back afterwards. The ``/tmp/alpino`` directory is
        removed with :func:`shutil.rmtree` before returning, and
        ``tempfile.tempdir`` is reset to ``None``. Callers should be aware
        that ``penn_output.txt`` is created in the working directory as a
        side effect, and that global ``tempfile`` state is temporarily
        modified.

        Parameters
        ----------
        sentences : list of str
            Dutch sentences to parse. Each element is parsed independently.

        Returns
        -------
        trees : list of nltk.tree.Tree
            One parse tree per sentence, in input order.

        Raises
        ------
        ImportError
            If ``alpinonaf`` is not available.
        """
        if not ALPINO_AVAILABLE:
            raise ImportError("alpinonaf required for Dutch parsing")
        
        trees = []
        tempfile.tempdir = '/tmp/alpino'
        
        if not os.path.exists('/tmp/alpino'):
            os.mkdir('/tmp/alpino')
        
        for sent in sentences:
            tmp = tempfile.NamedTemporaryFile(mode='wb', delete=False, prefix='alpino-')
            with open(tmp.name, 'wb') as fid:
                fid.write(bytes(sent, encoding='utf8'))
                fid.seek(0)
            with open(tmp.name, 'rb') as f:
                alpinonaf.parse(f, max_min_per_sent=5.0)
            with open('penn_output.txt', 'r') as f:
                penntree = f.read()
            trees.append(Tree.fromstring(penntree))
        
        shutil.rmtree('/tmp/alpino')
        tempfile.tempdir = None
        return trees
    
    def parse_text(self, text: str) -> List[Tree]:
        """Parse text into constituency trees.

        Tokenizes the text into sentences (using the language configured in
        ``config.language``) and parses each sentence with the parser named
        in ``config.parser_name``.

        Parameters
        ----------
        text : str
            Input text to parse, containing one or more sentences.

        Returns
        -------
        trees : list of nltk.tree.Tree
            One parse tree per sentence, in sentence order.

        Raises
        ------
        ValueError
            If ``config.parser_name`` is not ``"stanford"`` or ``"alpino"``.
        """
        if self.config.language == 'en':
            sentences = sent_tokenize(text, language='english')
        elif self.config.language == 'nl':
            sentences = sent_tokenize(text, language='dutch')
        elif self.config.language == 'fr':
            sentences = sent_tokenize(text, language='french')
        else:
            sentences = sent_tokenize(text, language=self.config.language)
        
        if self.config.parser_name == 'stanford':
            return self.get_stanford_tree(sentences)
        elif self.config.parser_name == 'alpino':
            return self.get_alpinopy_tree(sentences)
        else:
            raise ValueError(f"Unknown parser: {self.config.parser_name}")
    
    def depth_single_tree(self, tree: Tree, remove_S: bool = True, 
                           remove_unibranch_offset: bool = True) -> List[int]:
        """Get depth of each leaf in the parse tree.

        Computes the depth of every leaf (word) as the length of its
        tree position after collapsing unary productions. An offset can be
        subtracted to account for the root-level sentence node (``S``) and
        unary-branching nodes.

        Based on parseMetrics.py.

        Parameters
        ----------
        tree : nltk.tree.Tree
            Constituency parse tree.
        remove_S : bool
            Whether to subtract 1 from each depth to discount the root ``S``
            node. Default: ``True``.
        remove_unibranch_offset : bool
            Whether to subtract an additional 1 from each depth to discount
            unary-branching nodes collapsed by ``collapse_unary``. Default:
            ``True``.

        Returns
        -------
        depths : list of int
            Depth of each leaf, in leaf (word) order. Length equals the
            number of leaves in ``tree``.
        """
        tr = tree.copy(True)
        tr.collapse_unary(collapseRoot=True)
        offset = sum([remove_S, remove_unibranch_offset])
        return [len(pos) - offset for pos in tr.treepositions('leaves')]
    
    def opening_single_tree(self, tree: Tree) -> List[int]:
        """Get opening values for each leaf.

        For each leaf, counts how many branches open (i.e. how many sibling
        branches with index 0 precede it along its path from the root) after
        collapsing unary productions. Words earlier in the sentence tend to
        have higher opening values. The result is reversed for
        :meth:`closing_single_tree`.

        Based on parseMetrics.py.

        Parameters
        ----------
        tree : nltk.tree.Tree
            Constituency parse tree.

        Returns
        -------
        opening : list of int
            Opening value of each leaf, in leaf (word) order. Length equals
            the number of leaves in ``tree``.
        """
        tr = tree.copy(True)
        tr.collapse_unary(collapseRoot=True)
        pos = [tp for tp in tr.treepositions('leaves')]
        
        opening = []
        for p in pos:
            count = 0
            iterable = iter(p[-2::-1])
            for index in iterable:
                if index == 0:
                    count += 1
                else:
                    break
            opening.append(count)
        return opening
    
    def closing_single_tree(self, tree: Tree) -> List[int]:
        """Get closing values for each leaf.

        For each leaf, counts how many branches close at that leaf. Computed
        by reversing every subtree of the (unary-collapsed) tree and taking
        the reversed opening values, so words later in the sentence tend to
        have higher closing values.

        Based on parseMetrics.py.

        Parameters
        ----------
        tree : nltk.tree.Tree
            Constituency parse tree.

        Returns
        -------
        closing : list of int
            Closing value of each leaf, in leaf (word) order. Length equals
            the number of leaves in ``tree``.
        """
        tr = tree.copy(True)
        tr.collapse_unary(collapseRoot=True)
        for s in tr.subtrees():
            s.reverse()
        return self.opening_single_tree(tr)[::-1]
    
    def extract_from_tree(self, tree: Tree, features: List[str]) -> Dict[str, List[int]]:
        """Extract requested features from a single parse tree.

        Parameters
        ----------
        tree : nltk.tree.Tree
            Constituency parse tree.
        features : list of str
            Feature names to extract. Valid values are:

            - ``"depth"``: depth of each leaf via :meth:`depth_single_tree`
            - ``"opening"``: opening value of each leaf via :meth:`opening_single_tree`
            - ``"closing"``: closing value of each leaf via :meth:`closing_single_tree`
            - ``"tree_height"``: height of the whole tree, repeated once per leaf
            - ``"all"``: all of the above features

        Returns
        -------
        result : dict of str -> list of int
            Mapping from feature name to the per-leaf values, in leaf (word)
            order. Each list has one entry per leaf of ``tree``. Unknown
            feature names are silently ignored.
        """
        result = {}
        for feat in features:
            if feat == 'depth' or feat == 'all':
                result['depth'] = self.depth_single_tree(tree)
            if feat == 'opening' or feat == 'all':
                result['opening'] = self.opening_single_tree(tree)
            if feat == 'closing' or feat == 'all':
                result['closing'] = self.closing_single_tree(tree)
            if feat == 'tree_height' or feat == 'all':
                result['tree_height'] = [tree.height()] * len(tree.leaves())
        return result
    
    def extract(self, text: str, features: List[str] = None) -> Dict[str, List[int]]:
        """Extract syntactic features from text.

        Parses the text (see :meth:`parse_text`) and concatenates the
        per-tree features across all parse trees, so the returned lists
        contain one entry per word of the input text, in reading order.

        Parameters
        ----------
        text : str
            Input text to process.
        features : list of str, optional
            Feature names to extract. Valid values are:

            - ``"depth"``
            - ``"opening"``
            - ``"closing"``
            - ``"tree_height"``
            - ``"all"`` (extract all four features)

            If ``None``, defaults to ``["all"]``.

        Returns
        -------
        all_features : dict of str -> list of int
            Mapping from feature name to the per-word values, concatenated
            across all sentences in reading order. Empty dict if parsing
            produced no trees.
        """
        if features is None:
            features = ['all']
        
        trees = self.parse_text(text)
        if not trees:
            LOGGER.warning("No parse trees generated")
            return {}
        
        all_features = {}
        for tree in trees:
            tree_features = self.extract_from_tree(tree, features)
            for feat_name in tree_features:
                if feat_name not in all_features:
                    all_features[feat_name] = []
                all_features[feat_name].extend(tree_features[feat_name])
        
        return all_features
    
    def extract_to_dict(self, text: str, features: List[str] = None) -> Dict[str, Dict[int, float]]:
        """Extract features and return as word position -> value dict.

        Convenience wrapper around :meth:`extract` that converts each
        per-word value list into a mapping from word position (0-based index
        in the tokenized text) to the feature value as a float. Suitable for
        input to :meth:`pyeeg.features.alignment.AlignmentHandler.align_word_features`.

        Parameters
        ----------
        text : str
            Input text to process.
        features : list of str, optional
            Feature names to extract; see :meth:`extract` for valid values.
            If ``None``, defaults to ``["all"]``.

        Returns
        -------
        result : dict of str -> dict of int -> float
            Mapping from feature name to a dict of
            word position -> feature value, with one entry per word.
        """
        raw_features = self.extract(text, features)
        result = {}
        for feat_name, values in raw_features.items():
            result[feat_name] = {i: float(v) for i, v in enumerate(values)}
        return result
    
    def extract_to_array(self, text: str, features: List[str] = None) -> Tuple[List[str], np.ndarray]:
        """Extract features and return as arrays.

        Splits the input text on whitespace to obtain the word list, then
        extracts the requested features via :meth:`extract` and stacks them
        into a words-by-features matrix. Each column corresponds to one
        feature, in the order of ``features`` (or the canonical order
        ``["depth", "opening", "closing", "tree_height"]`` when ``"all"`` is
        requested).

        Parameters
        ----------
        text : str
            Input text to process.
        features : list of str, optional
            Feature names to extract; see :meth:`extract` for valid values.
            If ``None``, defaults to ``["all"]``.

        Returns
        -------
        words : list of str
            Words of the input text, in reading order, obtained by
            ``text.split()``. Note that this may not exactly match the leaf
            count of the parse trees if tokenization differs from
            whitespace splitting.
        feature_array : ndarray, shape (n_words, n_features)
            Feature values per word. Empty array (shape ``(0,)``) if no
            features were extracted.
        """
        words = text.split()
        feat_dict = self.extract(text, features)
        
        if not feat_dict:
            return words, np.array([])
        
        if features is None or 'all' in features:
            feature_names = ['depth', 'opening', 'closing', 'tree_height']
        else:
            feature_names = features
        
        n_words = len(list(feat_dict.values())[0]) if feat_dict else 0
        n_features = len(feature_names)
        feature_array = np.zeros((n_words, n_features))
        
        for i, feat_name in enumerate(feature_names):
            if feat_name in feat_dict:
                feature_array[:, i] = feat_dict[feat_name]
        
        return words, feature_array