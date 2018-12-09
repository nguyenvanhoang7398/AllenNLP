from readers.imdb.imdb_reader import compile_imdb, compile_imdb_pos_neg, ImdbDatasetReader

from allennlp.common import JsonDict
from allennlp.common.file_utils import cached_path
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
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 32

    """
    compile_imdb(TRAIN_POS_INPUT_DIR, TRAIN_POS_OUTPUT, 'pos')
    compile_imdb(TRAIN_NEG_INPUT_DIR, TRAIN_NEG_OUTPUT, 'neg')
    compile_imdb(TEST_POS_INPUT_DIR, TEST_POS_OUTPUT, 'pos')
    compile_imdb(TEST_NEG_INPUT_DIR, TEST_NEG_OUTPUT, 'neg')
    compile_imdb_pos_neg(TRAIN_POS_OUTPUT, TRAIN_NEG_OUTPUT, TRAIN_DATASET_PATH, RANDOM_SEED)
    compile_imdb_pos_neg(TEST_POS_OUTPUT, TEST_NEG_OUTPUT, TEST_DATASET_PATH, RANDOM_SEED)
    """

    reader = ImdbDatasetReader()
    train_dataset = reader.read(cached_path(TRAIN_DATASET_PATH))[:1000]
    dev_dataset = reader.read(cached_path(TEST_DATASET_PATH))[:100]
    # print(train_dataset[0]["tokens"], train_dataset[0]["label"])
    # raise Exception("Debugging")
    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    class LstmClassifier(Model):
        def __init__(self,
                     word_embeddings: TextFieldEmbedder,
                     encoder: Seq2VecEncoder,
                     vocab: Vocabulary) -> None:
            super().__init__(vocab)
            self.word_embeddings = word_embeddings
            self.encoder = encoder
            self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                              out_features=vocab.get_vocab_size('labels'))
            self.accuracy = CategoricalAccuracy()
            self.loss_function = torch.nn.CrossEntropyLoss()

        def forward(self,
                    tokens: Dict[str, torch.Tensor],
                    label: torch.Tensor = None) -> torch.Tensor:
            mask = get_text_field_mask(tokens)
            embeddings = self.word_embeddings(tokens)
            encoder_out = self.encoder(embeddings, mask)
            logits = self.hidden2tag(encoder_out)

            output = {"logits": logits}
            if label is not None:
                self.accuracy(logits, label)
                output["loss"] = self.loss_function(logits, label)

            return output
        def get_metrics(self, reset: bool = False) -> Dict[str, float]:
            return {"accuracy": self.accuracy.get_metric(reset)}

    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, lstm, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=10)
    trainer.train()

    @Predictor.register("sentence_classifier_predictor")
    class SentenceClassifierPredictor(Predictor):
        def __init__(self,
                     model: Model,
                     dataset_reader: DatasetReader) -> None:
            super().__init__(model, dataset_reader)

        def predict(self,
                    tokens: List[str]) -> JsonDict:
            return self.predict_json({"tokens": tokens})

        @overrides
        def _json_to_instance(self,
                              json_dict: JsonDict) -> Instance:
            tokens = json_dict["tokens"]
            return self._dataset_reader.text_to_instance(tokens)

    tokens = ['This', 'is', 'best', 'movie', 'ever', '!']
    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict(tokens)['logits']
    label_id = np.argmax(logits)
    print(model.vocab.get_token_from_index(label_id, 'labels'))
