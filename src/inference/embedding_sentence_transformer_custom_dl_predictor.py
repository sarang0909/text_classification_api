"""A Predictor module to load model and get prediction from sentence transformer custom dl model.

"""

import random
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.inference.predictor import Predictor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class EmbeddingSentenceTransformerCustomDlPredictor(Predictor):
    """A child class to load model and get output

    Args:
        Predictor (Predictor): Parent class

    """

    model = None

    def __init__(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        """A method to load model

        Returns:
            model: trained model
        """

        if self.model is None:
            self.model = Network(384, 4, 3)
            self.model.load_state_dict(
                torch.load(
                    "src/models/embedding_sentence_transformer_custom_dl/model.pth"
                )
            )

        return self.model

    def get_model_output(self, input_data):
        """A method to get model output from given text input
        Args:
            input_data (text):input text data

        Returns:
            output: model prediction
        """

        sentence_embeddings = torch.tensor(
            self.sent_transformer.encode([input_data])
        )
        model = self.get_model()
        model.eval()
        outputs = model(sentence_embeddings)
        return outputs.argmax().item()


class Network(torch.nn.Module):
    """A custom neural network class

    Args:
        vector_size (int): input embedding vector size
        hidden_units (int): number of hidden units
        num_classes (int): number of output classes
    """

    def __init__(self, vector_size, hidden_units, num_classes):
        """initializtion constructor

        Args:
            vector_size (int): input embedding vector size
            hidden_units (int): number of hidden units
            num_classes (int): number of output classes
        """

        super().__init__()
        # First fully connected layer
        self.fc1 = torch.nn.Linear(vector_size, hidden_units)
        # Second fully connected layer
        self.fc2 = torch.nn.Linear(hidden_units, num_classes)
        # Final output of sigmoid function
        self.output = torch.nn.Sigmoid()

    def forward(self, input_data):
        """Feedforward method

        Args:
            input : Input

        Returns:
            output: output
        """

        fc1 = self.fc1(input_data)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)
        # return output[:, -1]
        return output
