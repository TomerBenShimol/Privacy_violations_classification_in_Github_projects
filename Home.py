import streamlit as st

header = st.container()
approach = st.container()
goal = st.container()
datasets = st.container()
contributions = st.container()

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with header:
    st.title("Welcome to PVC")
    st.header("The world's first privacy violation classifier!")
    st.text(
        "Privacy violations occur when personal information is accessed or used without\nconsent and detecting them is important to prevent harm and protect privacy.\nIn this project, we trained machine-learning models that can accurately classify\nissues related to privacy violations."
    )

with approach:
    st.header("Approach")
    st.text(
        "We experimented with different text representation and classification models\nlike Support vector machine, eXtreme Gradient Booster, etc. and evaluated their\naccuracy on our dataset."
    )

with goal:
    st.header("Main goal")
    st.text(
        "Our findings can assist GitHub users in managing privacy issues more effectively.\nAn accurate classification could enhance trust and help protect sensitive\ninformation, benefitting social media and software development collaboration\nplatforms."
    )

with datasets:
    st.header("Datasets")
    st.text(
        "We collected and annotated a new dataset of issues reported in GitHub projects\nthat contains 981 issues. In addition, we worked with another dataset of privacy\nissues that contains 2556 issues collected and annotated by Jenny Guber."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="button1"><a href="/Dataset_1" class="link1" target="_self">Tomer & Eli</a></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="button1"><a href="/Dataset_2" class="link1" target="_self">Jenny Guber</a></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<p class="arrow">⬆️</p>', unsafe_allow_html=True)
    st.text(
        "You have two datasets to choose from, and we have trained six different models for\neach. Select one dataset and explore the models that have been trained on it."
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
