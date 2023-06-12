import streamlit as st
from functions import *
import pandas as pd
import time

header = st.container()
dataset = st.container()
dataset_statistics = st.container()
filler = st.container()
model = st.container()
text_input_container = st.empty()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with header:
    st.markdown(
        '<h1 class="main-header">Classification models on this page were trained on - "SCE dataset"</h1>',
        unsafe_allow_html=True,
    )

with dataset:
    st.markdown(
        '<h3 class="secondary-header">SCE dataset</h3><p class="description">Annotated by <a href="https://www.linkedin.com/in/tomerbenshimol/" class="link1">Tomer Ben Shimol</a> & <a href="https://www.linkedin.com/in/eli-amuyev-224a4b210/" class="link1">Eliyahu Amuyev</a><p>',
        unsafe_allow_html=True,
    )

with dataset_statistics:
    # Import data
    dataset = read_dataset1()
    st.write(dataset.head())
    st.markdown(
        f'<p class="description">This dataset contains {dataset.shape[0]} issues, of which {dataset.Classification.value_counts()[0]} are not related to privacy violations and {dataset.Classification.value_counts()[1]} are related to privacy violations. The word average is {avg_words(dataset)} and the character average is {avg_chars(dataset)}<p>',
        unsafe_allow_html=True,
    )

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
    option = st.selectbox(
        "Which text representation would you like to try?", text_representations
    )
    index1 = text_representations.index(option)
    if index1 < 5:
        option = st.selectbox("Which classifier would you like to try?", classifiers)
        index2 = classifiers.index(option)
    st.text("")
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

    model, metrics = load_model(1, model_name)
    st.text(metrics)
    st.text("* Please enter a description (text) of privacy related issue")
    text = st.text_input(
        "", placeholder="Write some text to classify...", label_visibility="collapsed"
    )
    result = -1
    if len(text) == 1:
        text_input_container.write(
            "There must be at least one word that is at least two characters long"
        )
    if text and len(text) > 1:
        st.write("")
        result = new_prediction(model, index1, 1, text)
        if result == 1:
            text_input_container.write("Privacy related! ⛔️")
        if result == 0:
            text_input_container.write("Non privacy related 🙏🏻")

with filler:
    st.text("")
