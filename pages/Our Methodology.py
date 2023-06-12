import streamlit as st
from functions import *

pipeline = st.container()
labeled_issues = st.container()
text_preprocessing = st.container()
text_representation = st.container()
classification = st.container()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with pipeline:
    st.markdown(
        '<h1 class="dataset1-main-header">Pipeline</h1>',
        unsafe_allow_html=True,
    )
    st.text("")
    st.image("img/pipeline.png")

with labeled_issues:
    st.markdown(
        f'</br><h3 class="dataset1-secondary-header">Labeled issues</h3><p class="description">We worked with two different datasets, one of them called ״SCE dataset״ arrived unclassified and we were required to classify it ourselves (separately) and the other called ״Haifa dataset״ came to us built including classification.<p>',
        unsafe_allow_html=True,
    )


with text_preprocessing:
    st.markdown(
        f'<h3 class="dataset1-secondary-header">Text Preprocessing</h3><p class="description">Each of the datasets underwent text processing through a function called clean_text and its purpose is to delete stop words, punctuation marks, Internet links, and HTML elements.</br></br>After that, keywords were extracted from the texts classified as privacy related from each dataset with the help of a tool called YAKE!</br></br>In addition, since the SCE dataset was unbalanced (773 issues, 565 unrelated to privacy and 208 related to privacy) we performed data augmentation using chatGPT and thus balanced it so that we could continue working (981 issues, 565 unrelated to privacy and 416 related to privacy).<p>',
        unsafe_allow_html=True,
    )

with text_representation:
    st.markdown(
        f'<h3 class="dataset1-secondary-header">Text Representation</h3><p class="bigger-bolder-p">TF-IDF<p><p class="description">TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to evaluate the importance of a term within a document in a collection or corpus. It aims to highlight words that are both frequently occurring within a document and relatively rare across the entire corpus.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-description"><b>Regular TF-IDF</b> - Normal use of this representation (as in the definition).<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-description"><b>Keywords X TF-IDF</b> - The feature names that were not included in the extracted keywords were omitted.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-description"><b>Normalized TF-IDF</b> - The weights of feature names included in the extracted keywords were multiplied by 2 and then vector normalization was performed.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-bolder-p">Word embeddings<p><p class="description">Word embedding refers to the representation of words in text analysis through real-valued vectors that encode the meaning of the words. This technique allows words with similar meanings to be closer to each other in a vector space. <p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-description"><b>fastText</b> - We obtained vectors of all feature names existing in the fastText dictionary and then calculated an average vector for each issue.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-description"><b>BERT</b> - BERT embeddings were used by preprocessing the data and encoding it with BERT preprocess and encoder layers, followed by the creation of a word embedding vector for each issue.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="bigger-bolder-p">Raw text<p><p class="description">Any string, block or group of only alphanumeric characters.<p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h3 class="dataset1-secondary-header">Classification models</h3><p class="description">SCE models were trained on the SCE dataset, we split the data as follows: 80% of training data & 20% of testing data (issues without labels).<p><p class="description">Haifa models were trained on the Haifa dataset, we split the data as follows: 70% of training data & 30% of testing data (issues without labels).<p><p class="description">You can choose between all the text representations mentioned above on the dedicated pages of the models (SCE or Haifa). It is also possible to see the evaluation results of each model according to each text representation.</p>',
        unsafe_allow_html=True,
    )
