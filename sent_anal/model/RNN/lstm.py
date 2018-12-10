from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy

import torch
from typing import Dict

@Model.register('lstm_classifier')
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