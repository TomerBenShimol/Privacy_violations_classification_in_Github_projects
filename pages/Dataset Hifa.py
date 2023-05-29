import streamlit as st
from functions import *
import pandas as pd
import time

header = st.container()
dataset_section = st.container()
model = st.container()
text_input_container = st.empty()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with header:
    st.markdown(
        '<h1 class="dataset1-main-header">This dataset was collected by researchers from University of Haifa</h1>',
        unsafe_allow_html=True,
    )

with dataset_section:
    # Import data
    dataset = read_dataset2()
    st.write(dataset.head())
    st.text(f"{dataset.shape[0]} Annotations.")
    st.text(f"{dataset.Classification.value_counts()[0]} Non-privacy violations.")
    st.text(f"{dataset.Classification.value_counts()[1]} Privacy violations.")
    st.text(f"Average words: {avg_words(dataset)}")
    st.text(f"Average characters: {avg_chars(dataset)}")
    st.text("Partition - 80% of training data & 20% of testing data.")
    st.text(
        "With the help of YAKE! we extracted keywords from the texts that were labeled as 1."
    )
    st.image("img/pv_d2.png")

with model:
    st.header("Model selection")
    text_representations = (
        "Regular TF-IDF",
        "Keywords x TF-IDF",
        "Normaized TF-IDF",
        "fastText",
        "BERT",
        "Raw text",
    )
    classifiers = ("eXtreme Gradient Booster", "Support vector machine")
    option = st.selectbox("Which text representation would you like to try?", text_representations)
    index1 = text_representations.index(option)
    if index1 < 5:
        option = st.selectbox("Which classifier would you like to try?", classifiers)
        index2 = classifiers.index(option)
    if index1 == 5:
        st.text("Classifier => BERT")

    if index1 < 5 and index2 == 0:
        model_name = "xgb" 
    elif index1 < 5 and index2 == 1:
        model_name = "svm" 
    elif index1 == 5:
        model_name = "bert_classifier"

    if model_name == "xgb" or model_name == "svm":
        model_name = f"{model_name}_model_{index1 + 1}.pickle"

    model, metrics = load_model(2, model_name)
    st.text("")
    st.text(metrics)
    st.text("* Please enter a description (text) of privacy violation.")
    text = st.text_input(
        "", placeholder="Write some text to classify..."
    )
    result = -1
    if len(text) == 1:
        text_input_container.write("There must be at least one word that is at least two characters long")
    if text and len(text) > 1:
        st.write("")
        result = new_prediction(model, index1, 2, text)
        if result == 1:
            text_input_container.write("Privacy Violation! ⛔️")
        if result == 0:
            text_input_container.write("Not a Privacy Violation 🙏🏻")
