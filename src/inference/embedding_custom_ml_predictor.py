"""A Predictor module to load model and get prediction from embedding custom ml model.

"""

import pickle
import random
import os
from transformers import DistilBertModel, AutoTokenizer
import numpy as np
import torch

from src.inference.predictor import Predictor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class EmbeddingCustomMLPredictor(Predictor):
    """A child class to load model and get output

    Args:
        Predictor (Predictor): Parent class

    """

    tuned_model = None

    def __init__(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.pre_trained_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_cls_embedding(self, input_data):
        """A method to get classification embeddings of sentence

        Args:
            input_data (str): input text data

        Returns:
            cls_embeddings: classification embeddings of sentence
        """
        tokenized_text = self.tokenizer(
            [input_data],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        outputs = self.pre_trained_model(**tokenized_text)
        last_hidden_states = outputs.last_hidden_state

        cls_embeddings = last_hidden_states[:, 0, :].detach().numpy()
        return cls_embeddings

    def get_model(self):
        """A method to load model

        Returns:
            model: trained model
        """

        if self.tuned_model is None:
            with open(
                "src/models/embedding_custom_ml.pkl", "rb"
            ) as model_file:
                self.tuned_model = pickle.load(model_file)
        return self.tuned_model

    def get_model_output(self, input_data):
        """A method to get model output from given text input
        Args:
            input_data (text):input text data

        Returns:
            output: model prediction
        """

        model = self.get_model()
        input_embedding = self.get_cls_embedding(input_data)
        return model.predict(input_embedding)[0]
