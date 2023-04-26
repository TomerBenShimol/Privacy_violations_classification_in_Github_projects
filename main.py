import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
approach = st.container()
goal = st.container()

with header:
	st.title('Welcome to PVC')
	st.header("The world's first privacy violation classifier!")
	st.text("Privacy violations occur when personal information is accessed or used without\nconsent and detecting them is important to prevent harm and protect privacy.\nIn this project, we trained machine-learning models that can accurately classify\nissues related to privacy violations.")


with dataset:
	st.header("Dataset")
	st.text("We collected and annotated a new dataset of issues reported in GitHub projects.")
	# Import data 
	dataset = pd.read_csv("../datasets/dataset-mixed-classifications.csv", encoding = "ISO-8859-1")

	# Ignoring unnecessary columns 
	dataset.drop("ï»¿URL", axis=1, inplace=True)
	dataset.drop("ID", axis=1, inplace=True)
	dataset.drop("Classification - Tomer", axis=1, inplace=True)
	dataset.drop("Classification - Eli", axis=1, inplace=True)
	dataset.drop("Disagreements - Red", axis=1, inplace=True)
	dataset.dropna(subset="Classification", inplace=True)
	dataset = dataset.sample(frac=1)
	st.write(dataset.head())

with approach:
	st.header("Approach")
	st.text("We experimented with different text representation and classification models\nlike Support vector machine, eXtreme Gradient Booster, etc. and evaluated their\naccuracy on our dataset.")


with goal:
	st.header('Main goal')
	st.text("Our findings can assist GitHub users in managing privacy issues more effectively.\nAn accurate classification could enhance trust and help protect sensitive\ninformation, benefitting social media and software development collaboration\nplatforms.")

