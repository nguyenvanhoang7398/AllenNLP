from readers.imdb.imdb_reader import compile_imdb, compile_imdb_pos_neg, ImdbDatasetReader

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.predictors import Predictor
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.trainer import Trainer

import numpy as np
from overrides import overrides
import torch
from typing import Dict, List

if __name__ == "__main__":
    TRAIN_POS_INPUT_DIR = "data/aclImdb/train/pos"
    TRAIN_NEG_INPUT_DIR = "data/aclImdb/train/neg"
    TEST_POS_INPUT_DIR = "data/aclImdb/test/pos"
    TEST_NEG_INPUT_DIR = "data/aclImdb/test/neg"
    TRAIN_POS_OUTPUT = "data/aclImdb/train/imdb_train_pos.txt"
    TRAIN_NEG_OUTPUT = "data/aclImdb/train/imdb_train_neg.txt"
    TEST_POS_OUTPUT = "data/aclImdb/test/imdb_test_pos.txt"
    TEST_NEG_OUTPUT = "data/aclImdb/test/imdb_test_neg.txt"
    TRAIN_DATASET_PATH = "data/aclImdb/train/imdb_train.txt"
    TEST_DATASET_PATH = "data/aclImdb/test/imdb_test.txt"
    RANDOM_SEED = 7

    compile_imdb(TRAIN_POS_INPUT_DIR, TRAIN_POS_OUTPUT, 'pos')
    compile_imdb(TRAIN_NEG_INPUT_DIR, TRAIN_NEG_OUTPUT, 'neg')
    compile_imdb(TEST_POS_INPUT_DIR, TEST_POS_OUTPUT, 'pos')
    compile_imdb(TEST_NEG_INPUT_DIR, TEST_NEG_OUTPUT, 'neg')
    compile_imdb_pos_neg(TRAIN_POS_OUTPUT, TRAIN_NEG_OUTPUT, TRAIN_DATASET_PATH, RANDOM_SEED)
    compile_imdb_pos_neg(TEST_POS_OUTPUT, TEST_NEG_OUTPUT, TEST_DATASET_PATH, RANDOM_SEED)

    reader = ImdbDatasetReader()
    train_dataset = reader.read(TRAIN_DATASET_PATH)
    dev_dataset = reader.read(TEST_DATASET_PATH)
