import pandas as pd
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Cleans a given string by removing URLs, HTML elements, punctuations, stop words, and extra white spaces.
def clean_text(
    string: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
    stop_words=['the', 'a', 'and', 'is', 'be', 'will', 'steps', 'to', 'reproduce']) -> str:
    """
    A method to clean text 
    """
    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)
    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)
    # Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
    # Converting the text to lower
    string = string.lower()
    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])
    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()
    return string 

def read_dataset1():
 dataset = pd.read_csv("datasets/dataset-mixed-classifications.csv", encoding = "ISO-8859-1")

 # Ignoring unnecessary columns 
 dataset.drop("ï»¿URL", axis=1, inplace=True)
 dataset.drop("ID", axis=1, inplace=True)
 dataset.drop("Classification - Tomer", axis=1, inplace=True)
 dataset.drop("Classification - Eli", axis=1, inplace=True)
 dataset.drop("Disagreements - Red", axis=1, inplace=True)
 dataset.dropna(subset="Classification", inplace=True)
 ## Concat the Title column to the Body column into a new column - Text  
 dataset["Text_"] = dataset["Title"].astype(str) + " " + dataset["Body"].astype(str)
 # The only columns we are interested in are: 'Classification' & 'Text'
 dataset['Title'] = dataset['Text_']
 dataset.rename(columns={"Title": "Text"}, inplace=True)
 dataset.drop("Text_", axis=1, inplace=True)
 dataset.drop("Body", axis=1, inplace=True)
 # Text preprocessing
 for i in range(dataset.Text.shape[0]):
  dataset.loc[i, 'Text'] = clean_text(dataset.loc[i, 'Text'])
 dataset = dataset.dropna(subset="Classification").loc[dataset.Classification != 3]
 dataset_pv_augmentation = pd.read_csv("datasets/data_augmentation_privacy_violations.csv", encoding = "ISO-8859-1")
 dataset_pv_augmentation.drop("Unnamed: 0", axis=1, inplace=True)
 # Text preprocessing
 for i in range(dataset_pv_augmentation.Text.shape[0]):
  dataset_pv_augmentation.loc[i, 'Text'] = clean_text(dataset_pv_augmentation.loc[i, 'Text'])
 # Concat & shuffle
 dataset = pd.concat([dataset, dataset_pv_augmentation])
 dataset = dataset.sample(frac=1)
 return dataset

def read_dataset2():
	dataset = pd.read_csv("datasets/jenny-dataset.csv", encoding = "ISO-8859-1")
	# Ignoring unnecessary columns 
	dataset.drop("ï»¿Dataset ID", axis=1, inplace=True)
	dataset.drop("Issue ID", axis=1, inplace=True)
	dataset.drop("Source", axis=1, inplace=True)
	dataset = dataset.sample(frac=1)
	return dataset

def load_model(id, name):
 if id == 1:
  return pickle.load(open(f'models/tomer_and_eli/{name}', "rb"))
 if id == 2:
  return pickle.load(open(f'models/jenny/{name}', "rb"))

def load_feature_names(id):
 if id == 1:
  return np.load(f'feature_names/tomer_and_eli/feature_names.npy', allow_pickle=True)
 if id == 2:
  return np.load(f'feature_names/jenny/feature_names.npy', allow_pickle=True)

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

def new_prediction(model, text_to_predict, feature_names):
    preds = preprocess_text_for_predict(text_to_predict, feature_names)
    return model.predict(preds)[0]
