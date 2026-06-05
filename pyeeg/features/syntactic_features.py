"""
Syntactic Feature Extraction

This module provides functionality for extracting syntactic features from
constituency trees, incorporating code from process_txt/pyProcess/parseMetrics.py.

Features:
- Depth: Depth of each node in the parse tree
- Opening nodes: Number of branches opening at each leaf
- Closing nodes: Number of branches closing at each leaf

Based on parseMetrics.py by Hugo Weissbart
"""

import os
import logging
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from nltk.tree import Tree
    from nltk.parse.stanford import StanfordParser
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import alpinonaf
    ALPINO_AVAILABLE = True
except ImportError:
    ALPINO_AVAILABLE = False


@dataclass
class ParserConfig:
    """Configuration for external parser."""
    parser_name: str = "stanford"
    parser_path: Optional[str] = None
    model_path: Optional[str] = None
    language: str = "en"
    timeout: int = 30


class SyntacticFeatureExtractor:
    """
    Extract syntactic features from text using constituency parsing.
    Based on: process_txt/pyProcess/parseMetrics.py by Hugo Weissbart
    """
    
    def __init__(self, config: ParserConfig = None):
        if config is None:
            config = ParserConfig()
        self.config = config
    
    def get_stanford_tree(self, sentences: List[str], path_to_jar: Optional[str] = None) -> List[Tree]:
        """Parse sentences using Stanford Parser. Based on parseMetrics.py"""
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK required for Stanford parsing")
        
        logger.info("Loading Stanford Parser...")
        
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
        logger.info("Stanford Parser loaded")
        
        parse = parser.raw_parse_sents(sentences)
        trees = []
        for treelist in parse:
            for tree in treelist:
                trees.append(tree)
        return trees
    
    def get_alpinopy_tree(self, sentences: List[str]) -> List[Tree]:
        """Parse Dutch sentences using Alpino. Based on parseMetrics.py"""
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
        """Parse text into constituency trees."""
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
        """Get depth of each leaf in the parse tree. Based on parseMetrics.py"""
        tr = tree.copy(True)
        tr.collapse_unary(collapseRoot=True)
        offset = sum([remove_S, remove_unibranch_offset])
        return [len(pos) - offset for pos in tr.treepositions('leaves')]
    
    def opening_single_tree(self, tree: Tree) -> List[int]:
        """Get opening values for each leaf. Based on parseMetrics.py"""
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
        """Get closing values for each leaf. Based on parseMetrics.py"""
        tr = tree.copy(True)
        tr.collapse_unary(collapseRoot=True)
        for s in tr.subtrees():
            s.reverse()
        return self.opening_single_tree(tr)[::-1]
    
    def extract_from_tree(self, tree: Tree, features: List[str]) -> Dict[str, List[int]]:
        """Extract requested features from a single parse tree."""
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
        """Extract syntactic features from text."""
        if features is None:
            features = ['all']
        
        trees = self.parse_text(text)
        if not trees:
            logger.warning("No parse trees generated")
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
        """Extract features and return as word position -> value dict."""
        raw_features = self.extract(text, features)
        result = {}
        for feat_name, values in raw_features.items():
            result[feat_name] = {i: float(v) for i, v in enumerate(values)}
        return result
    
    def extract_to_array(self, text: str, features: List[str] = None) -> Tuple[List[str], np.ndarray]:
        """Extract features and return as arrays."""
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