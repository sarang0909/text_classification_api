"""A Predictor module to load model and get prediction from embedding
   hugging face trainer api model.
"""

from src.inference.predictor import Predictor
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
import random
import numpy as np
import torch

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class EmbeddingHuggingFacePredictor(Predictor):
    """A child class to load model and get output

    Args:
        Predictor (Predictor): Parent class

    """

    model = None

    def __init__(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        """A method to load model

        Returns:
            model: trained model
        """

        if self.model is None:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "src/models/embedding_hugging_face"
            )
        return self.model

    def get_model_output(self, input_data):
        """A method to get model output from given text input
        Args:
            input_data (text):input text data

        Returns:
            output: model prediction
        """

        tokenized_text = self.tokenizer(
            [input_data],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        model = self.get_model()
        outputs = model(**tokenized_text)
        return outputs.logits.argmax().item()
