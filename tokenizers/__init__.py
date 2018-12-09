import os

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}

from .corenlp_tokenizer import CoreNLPTokenizer
