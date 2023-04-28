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
        '<h1 class="dataset1-main-header">This dataset was annotated by</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2 class="dataset1-secondary-header"><a href="https://www.linkedin.com/in/jennyguber/" class="link1">Jenny Guber</a></h2>',
        unsafe_allow_html=True,
    )

with dataset_section:
    # Import data
    dataset = read_dataset2()
    st.write(dataset.head())
    st.text(f"{dataset.shape[0]} Annotations.")
    st.text(f"{dataset.Classification.value_counts()[0]} Non-privacy violations.")
    st.text(f"{dataset.Classification.value_counts()[1]} Privacy violations.")
    st.text("* After performing data augmentation - with the help of chatGPT.")
    st.text(
        "With the help of YAKE! we extracted keywords from the texts that were labeled as 1."
    )
    st.image("img/pv_d2.png")

with model:
    st.header("Model selection")
    options1 = (
        "Regular TF-IDF",
        "Keywords x TF-IDF",
        "Normaized TF-IDF",
        "fastText",
        "BERT",
        "BERT Classifier",
    )
    options2 = ("eXtreme Gradient Booster", "Support vector machine")
    option1 = st.selectbox("Which model would you like to try?", options1)
    index1 = options1.index(option1)
    if index1 != 5:
        option2 = st.selectbox("Classifier?", options2)
        index2 = options2.index(option2)
    else:
        index2 = None
    st.text(
        "* Please enter text that describes for you a type of privacy violation\n  in software."
    )
    text = text_input_container.text_input(
        "", placeholder="Write some text to classify..."
    )
    model_name = "xgb"
    if index2 and index2 == 1:
        model_name = "svm"
    model_name = f"{model_name}_model_{index1 + 1}.pickle"
    if index1 == 5:
        model_name = "bert_classifier"
    model = load_model(2, model_name)
    #  feature_names = load_feature_names(1)
    result = -1
    if text:
        text_input_container.write("")
        result = new_prediction(model, index1, 2, text)
        if result == 1:
            text_input_container.write("Privacy Violation! ⛔️")
            time.sleep(3)
            if "pv" in st.session_state:
                st.session_state["pv"] = ""
            text = text_input_container.text_input(
                "", placeholder="Write some text to classify...", key="pv"
            )
        if result == 0:
            text_input_container.write("Not a Privacy Violation 🙏🏻")
            time.sleep(3)
            if "!pv" in st.session_state:
                st.session_state["!pv"] = ""
            text = text_input_container.text_input(
                "", placeholder="Write some text to classify...", key="!pv"
            )
