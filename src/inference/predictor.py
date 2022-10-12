"""A Predictor abstract module to load models and get their predictions.

"""
from abc import ABC, abstractmethod


class Predictor(ABC):
    """
    This is an abstract class for model load and prediction methods.

    """

    @abstractmethod
    def get_model(self):
        """An abstract method to get and load  model"""

    @abstractmethod
    def get_model_output(self, input_data):
        """An abstract method to get model output

        Args:
            input_data (str): Input text data to model
        """

    def map_output(self, output):
        """A method to map outputs from different models to common output format
        Args:
            output (str): model output
        Returns:
            str: common output format
        """
        outputs = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE",
            "NEGATIVE": "NEGATIVE",
            "NEUTRAL": "NEUTRAL",
            "POSITIVE": "POSITIVE",
        }
        return outputs[output]
