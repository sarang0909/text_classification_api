"""A factory module to get predictor class based on type"""
from src.inference.embedding_custom_dl_predictor import (
    EmbeddingCustomDLPredictor,
)

from src.inference.embedding_custom_ml_predictor import (
    EmbeddingCustomMLPredictor,
)
from src.inference.embedding_hugging_face_predictor import (
    EmbeddingHuggingFacePredictor,
)
from src.inference.tfidf_pycaret_predictor import PycaretPredictor
from src.inference.tfidf_custom_ml_predictor import TfIdfCustomMlPredictor
from src.inference.tfidf_custom_dl_keras_predictor import (
    TfIdfCustomDlKerasPredictor,
)
from src.inference.embedding_sentence_transformer_custom_dl_predictor import (
    EmbeddingSentenceTransformerCustomDlPredictor,
)


def get_predictor(model_type):
    """A method to retun Predictor class object

    Args:
        model_type (str): Model type

    Returns:
        Predictor: A predictor class
    """
    if model_type is None:
        return None
    elif model_type == "tfidf_pycaret":
        return PycaretPredictor()
    elif model_type == "tfidf_custom_ml":
        return TfIdfCustomMlPredictor()
    elif model_type == "tfidf_custom_dl_keras":
        return TfIdfCustomDlKerasPredictor()
    elif model_type == "embedding_custom_ml":
        return EmbeddingCustomMLPredictor()
    elif model_type == "embedding_custom_dl":
        return EmbeddingCustomDLPredictor()
    elif model_type == "embedding_hugging_face":
        return EmbeddingHuggingFacePredictor()
    elif model_type == "embedding_sentence_transformer_custom_dl":
        return EmbeddingSentenceTransformerCustomDlPredictor()
