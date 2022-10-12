"""A main script to run streamlit application.

"""
import streamlit as st
import pandas as pd
from src.utility.loggers import logger
from src.inference.predictor_factory import get_predictor

# st.set_page_config(layout="wide")
st.title("Text Classification")
st.text("Models are trained on small sample of news data")
st.text("to identify POSITIVE/NEUTRAL/NEGATIVE sentence/paragraph")
data = [
    ("TF-IDF", "Best model from pycaret", "Pycaret", "tfidf_pycaret"),
    (" TF-IDF", "ML model by experiments", "sklearn", "tfidf_custom_ml"),
    (" TF-IDF", "Custom neural network", "Keras", "tfidf_custom_dl_keras"),
    (
        "Distilbert embeddings",
        "ML model by experiments",
        "sklearn,transformers",
        "embedding_custom_ml",
    ),
    (
        "Distilbert embeddings",
        "Custom neural network",
        "Pytorch,transformers",
        "embedding_custom_dl",
    ),
    (
        "Distilbert embeddings",
        "transformers neural network",
        "Pytorch,transformers",
        "embedding_hugging_face",
    ),
    (
        "sentence transformer embeddings",
        "Custom neural network",
        "Pytorch,sentence_transformer",
        "embedding_sentence_transformer_custom_dl",
    ),
]
df = pd.DataFrame(
    data, columns=[" Input vectors", "Model", "Library", "Model Name"]
)
st.dataframe(df, use_container_width=False)
form = st.form(key="my-form")
input_data = form.text_area("Enter text for classification")

classification_method = form.radio(
    "Choose a text classification method",
    (
        "tfidf_pycaret",
        "tfidf_custom_ml",
        "tfidf_custom_dl_keras",
        "embedding_custom_ml",
        "embedding_sentence_transformer_custom_dl",
    ),
)
submit = form.form_submit_button("Submit")


if submit:
    try:
        predictor = get_predictor(classification_method)

        output = predictor.map_output(predictor.get_model_output(input_data))
        st.write("model_selected:", classification_method)
        st.write("model_output:", output)
    except Exception as error:
        message = "Error while creating output"
        logger.error(message, str(error))
