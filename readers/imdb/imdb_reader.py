from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import logging
from os import listdir
from os.path import isfile, join
from overrides import overrides
import random
from tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from tokenizers.tokenizer import Tokenizer
from typing import Dict, List

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

def compile_imdb(input_dir, output_file, label, delimiter='\t'):
    input_files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    out_file = open(output_file, 'w')
    for input_file in input_files:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                line_with_label = delimiter.join([line, label])
                out_file.write(line_with_label + "\n")
    out_file.close()

def compile_imdb_pos_neg(input_pos, input_neg, output_file, random_seed=None):
    dataset = []
    for input_file in [input_pos, input_neg]:
        with open(input_file, 'r') as f:
            for line in f.readlines():
                dataset.append(line)
    if random_seed:
        random.Random(random_seed).shuffle(dataset)
    with open(output_file, 'w') as f:
        for data in dataset:
            f.write(data)

class ImdbDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 delimiter: str = '\t',
                 lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers  or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer or CoreNLPTokenizer()
        self.delimiter = delimiter

    @overrides
    def _read(self, file_path):
        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                split_line = line.split(self.delimiter)
                review = line[0] if len(line) > 0 else None
                label = line[1] if len(line) > 1 else None
                tokenized_review = self.tokenizer.tokenize(review).words() if review is not None else []
                instance = self.text_to_instance(tokenized_review, label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         sentiment: str = None) -> Instance:
        review_field = TextField([Token(x) for x in tokens], self.token_indexers)
        fields: Dict[str, Field] = {"tokens": review_field}

        if sentiment is not None:
            fields['label'] = LabelField(sentiment)
        return Instance(fields)
