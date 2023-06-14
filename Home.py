import streamlit as st

header = st.container()
approach = st.container()
goal = st.container()
datasets = st.container()
models = st.container()
contributions = st.container()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with header:
    st.title("Welcome to PVC")
    st.header("The world's first privacy issues classifier!")
    st.text(
        "Privacy violations occur when personal information is accessed or used without\nconsent and detecting them is important to prevent harm and protect privacy.\nIn this project, we trained machine-learning models that can accurately classify\nissues related to privacy."
    )

with approach:
    st.header("Approach")
    st.text(
        "We experimented with different text representation and classification models\nlike Support vector machine, eXtreme Gradient Booster, etc. and evaluated their\naccuracy on two different datasets."
    )

with goal:
    st.header("Main goal")
    st.text(
        "Our findings can assist GitHub users in managing privacy issues more effectively.\nAn accurate classification could enhance trust and help protect sensitive\ninformation, benefitting social media and software development collaboration\nplatforms."
    )

with datasets:
    st.header("Datasets")
    st.text(
        "We worked with two distinct sets of data. The first one was initially unclassified,\nand we took the responsibility of classifying it ourselves (referred to as the SCE\ndataset). The second dataset was created by researchers from the University of\nHaifa and was already classified when it was provided to us (known as the Haifa\ndataset)."
    )

with models:
    st.header("Models")
    st.text(
        "In the end we were able to train 9 different models for each of the datasets.\nFor each model, metrics and performance are different depending on the text\nrepresentation."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="button1"><a href="/SCE_models" class="link1" target="_self">SCE models</a></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="button1"><a href="/Hifa_models" class="link1" target="_self">Haifa models</a></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="arrow">⬆️</p>', unsafe_allow_html=True)
    st.text(
        "You have two configurations to choose from. Select one model configuration\nand explore our models."
    )


with contributions:
    st.header("Contributions")
    st.text(
        "Would you like to contribute to our project by uploading your own datasets\nof privacy-related issues and their classifications?"
    )
    st.markdown(
        '<div class="button2"><a href="/Contributions" class="link1" target="_self">Upload your own dataset</a></div>',
        unsafe_allow_html=True,
    )
