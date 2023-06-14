import streamlit as st
import os

header = st.container()
join_us = st.container()
upload_file = st.container()


def is_csv(dataset):
    if dataset.type != "text/csv":
        return False
    return True


with header:
    st.header("Join us in improving privacy protection - Contribute your datasets!")

with join_us:
    st.text(
        "Hey there! We're working on a project to develop machine-learning models that\ncan accurately classify issues related to privacy violations on GitHub projects.\nTo make our model more accurate, we need your help! If you have encountered any\nprivacy-related issues, you can upload your own datasets as a csv file that\ncontains two columns - Text & Classification.\n\nText - description of the issue.\n\nClassification - 1 for privacy related issue and 0 for non-privacy related issue.\n\nBy doing so, you can contribute to expanding the size and diversity of our dataset,\nwhich can help us to identify more patterns and improve the accuracy of our model.\nAdditionally, your contributions can help us to identify new types of privacy\nrelated issues that may not have been captured in our initial dataset. Thank you\nfor your help in making our project better and more effective in protecting\nsensitive information!"
    )

with upload_file:
    dataset = st.file_uploader(
        "", accept_multiple_files=False, key="contributions_input"
    )
    if dataset:
        valid = is_csv(dataset)
        if valid:
            with open(os.path.join("contributions", dataset.name), "wb") as f:
                f.write(dataset.getbuffer())
            st.text("File uploaded successfully, Thank you!")
        else:
            st.text("File must be of type csv!")
