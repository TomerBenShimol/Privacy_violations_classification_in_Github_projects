import streamlit as st
import os

header = st.container()
join_us = st.container()
upload_file = st.container()

with header:
    st.header("Join us in improving privacy protection - Contribute your datasets!")

with join_us:
    st.text(
        "Hey there! We're working on a project to develop machine-learning models that\ncan accurately classify issues related to privacy violations on GitHub projects.\nTo make our model more accurate, we need your help! If you have encountered any\nprivacy-related issues, you can upload your own datasets as a csv file that\ncontains two columns - Text & Classification.\n\nText - description of the issue.\n\nClassification - 1 for privacy violation and 0 for non-privacy violation.\n\nBy doing so, you can contribute to expanding the size and diversity of our dataset,\nwhich can help us to identify more patterns and improve the accuracy of our model.\nAdditionally, your contributions can help us to identify new types of privacy\nviolations that may not have been captured in our initial dataset. Thank you for\nyour help in making our project better and more effective in protecting sensitive\ninformation!"
    )

with upload_file:
    dataset = st.file_uploader("")
    if dataset:
        with open(os.path.join("contributions", dataset.name), "wb") as f:
            f.write(dataset.getbuffer())
