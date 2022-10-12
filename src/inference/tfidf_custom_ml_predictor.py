"""A Predictor module to load model and get prediction from tf-idf custom ml model.

"""
import pickle
import pandas as pd
from src.inference.predictor import Predictor


class TfIdfCustomMlPredictor(Predictor):
    """A child class to load model and get output

    Args:
        Predictor (Predictor): Parent class

    """

    model = None
    tf_idf_vectorizer = None

    def get_model(self):
        """A method to load model

        Returns:
            model: trained model
        """
        if self.model is None:
            with open("src/models/tfidf_custom_ml.pkl", "rb") as model_file:
                self.model = pickle.load(model_file)
        return self.model

    def get_model_output(self, input_data):
        """A method to get model output from given text input
        Args:
            input_data (text):input text data

        Returns:
            output: model prediction
        """

        tf_idf = self.get_tf_idf_vectorizer()
        model = self.get_model()
        test_input = pd.DataFrame(
            tf_idf.transform([input_data]).toarray(),
            columns=tf_idf.get_feature_names(),
        )

        return model.predict(test_input)[0]

    def get_tf_idf_vectorizer(self):
        """a method to load tf idf vectorizer model

        Returns:
            tf_idf vectorizer: tf_idf vectorizer
        """

        if self.tf_idf_vectorizer is None:
            with open("src/models/tfidf_vectorizer_custom_ml.pkl", "rb") as file:
                self.tf_idf_vectorizer = pickle.load(file)
        return self.tf_idf_vectorizer
