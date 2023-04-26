import streamlit as st
import dataset as dt
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

header = st.container()
dataset = st.container()
approach = st.container()
goal = st.container()
classifier = st.container()


# Prepering the text for model prediction
def preprocess_text_for_predict(text_to_process, feature_names):
    if type(text_to_process) is str:
        text_to_process = pd.Series(text_to_process)
    # Create an array of zeros
    data = np.zeros((1, len(feature_names)))
    text = pd.DataFrame(data,index=[text_to_process], columns=feature_names)

    # TF-IDF preprocessing
    new_tfidf = text_to_process
    # Initialize a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(new_tfidf.values.astype('U'))
    new_tfidf = pd.DataFrame(vector.toarray(), index=new_tfidf.tolist(), columns=vectorizer.get_feature_names_out())
    weights = []

    # Saving tfidf original values
    for i in range(new_tfidf.shape[0]):
        for j in range(new_tfidf.shape[1]):
            weights.append(new_tfidf.iloc[i][j])

    # new_tfidf feature names
    columns = new_tfidf.columns.tolist()

    # If they have common feature vector the tfidf value changes to new_tfidf value, else it remains zero
    for i in range(len(columns)):
        if columns[i] in text.columns:
            index = list(text.columns).index(columns[i])
            text.iloc[0][index] = weights[i]
    return text


with header:
	st.title('Welcome to PVC')
	st.header("The world's first privacy violation classifier!")
	st.text("Privacy violations occur when personal information is accessed or used without\nconsent and detecting them is important to prevent harm and protect privacy.\nIn this project, we trained machine-learning models that can accurately classify\nissues related to privacy violations.")

with dataset:
	st.header("Dataset")
	st.text("We collected and annotated a new dataset of issues reported in GitHub projects.")
	# Import data 
	dataset = dt.read_dataset()
	st.write(dataset.head())

with approach:
	st.header("Approach")
	st.text("We experimented with different text representation and classification models\nlike Support vector machine, eXtreme Gradient Booster, etc. and evaluated their\naccuracy on our dataset.")


with goal:
	st.header('Main goal')
	st.text("Our findings can assist GitHub users in managing privacy issues more effectively.\nAn accurate classification could enhance trust and help protect sensitive\ninformation, benefitting social media and software development collaboration\nplatforms.")

with classifier:
	st.header("Try PVC")
	st.text("* Please enter text that describes for you a type of privacy violation\n  in software.")
	text = st.text_input("", placeholder="Write some text...")
	model = pk.load(open(f"/Users/tomerbenshimol/Desktop/Software_Engineering_Project/models/jenny/xgb_model_1.pickle", "rb"))
	feature_names = np.load("/Users/tomerbenshimol/Desktop/Software_Engineering_Project/feature_names/feature_names_Jenny.npy", allow_pickle=True)
	if text != '':
		text = preprocess_text_for_predict(text, feature_names)
		result = model.predict(text)[0]
		if result == 1:
			st.write('Privacy Violation!')
		if result == 0:
			st.write('Not a Privacy Violation ;)')
