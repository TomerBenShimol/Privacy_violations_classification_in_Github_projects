import pandas as pd
import streamlit as st
import re
import pickle
import numpy as np
import tensorflow as tf
import fasttext
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
import pymongo
from pymongo.server_api import ServerApi


# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(
        "mongodb+srv://tomerbe3:R584tcs8L3NPZlj9@cluster0.taqvwiw.mongodb.net/?retryWrites=true&w=majority",
        server_api=ServerApi("1"),
    )


client = init_connection()


def get_DB():
    return client


metrics = {
    "dataset1": {
        "xgb_model_1": "Confusion matrix:\n[95 18]\n[30 54]\n\nAccuracy: 75.63%\nPrecision: 75.00%\nRecall: 64.29%\nF1 Score: 69.23%",
        "svm_model_1": "Confusion matrix:\n[101  12]\n[24   60]\n\nAccuracy: 81.73%\nPrecision: 83.33%\nRecall: 71.43%\nF1 Score: 76.92%",
        "xgb_model_2": "Confusion matrix:\n[95 18]\n[39 45]\n\nAccuracy: 71.07%\nPrecision: 71.43%\nRecall: 53.57%\nF1 Score: 61.22%",
        "svm_model_2": "Confusion matrix:\n[92 21]\n[36 48]\n\nAccuracy: 71.07%\nPrecision: 69.57%\nRecall: 57.14%\nF1 Score: 62.75%",
        "xgb_model_3": "Confusion matrix:\n[93 20]\n[29 55]\n\nAccuracy: 75.13%\nPrecision: 73.33%\nRecall: 65.48%\nF1 Score: 69.18%",
        "svm_model_3": "Confusion matrix:\n[101  12]\n[21   63]\n\nAccuracy: 83.25%\nPrecision: 84.00%\nRecall: 75.00%\nF1 Score: 79.25%",
        "xgb_model_4": "Confusion matrix:\n[94 19]\n[24 60]\n\nAccuracy: 78.17%\nPrecision: 75.95%\nRecall: 71.43%\nF1 Score: 73.62%",
        "svm_model_4": "Confusion matrix:\n[91 22]\n[20 64]\n\nAccuracy: 78.68%\nPrecision: 74.42%\nRecall: 76.19%\nF1 Score: 75.29%",
        "xgb_model_5": "Confusion matrix:\n[78 35]\n[39 45]\n\nAccuracy: 62.44%\nPrecision: 56.25%\nRecall: 53.57%\nF1 Score: 54.88%",
        "svm_model_5": "Confusion matrix:\n[101  12]\n[71 13]\n\nAccuracy: 57.87%\nPrecision: 52.00%\nRecall: 15.48%\nF1 Score: 23.85%",
        "bert_classifier": "Loss: 66.99%\nAccuracy: 58.88%\n\nPrecision: 55.17%\nRecall: 19.05%",
    },
    "dataset2": {
        "xgb_model_1": "Confusion matrix:\n[253   0]\n[  1 258]\n\nAccuracy: 99.80%\nPrecision: 100.00%\nRecall: 99.61%\nF1 Score: 99.81%",
        "svm_model_1": "Confusion matrix:\n[250   3]\n[  2 257]\n\nAccuracy: 99.02%\nPrecision: 98.85%\nRecall: 99.23%\nF1 Score: 99.04%",
        "xgb_model_2": "Confusion matrix:\n[232   21]\n[  52 207]\n\nAccuracy: 85.74%\nPrecision: 90.79%\nRecall: 79.92%\nF1 Score: 85.01%",
        "svm_model_2": "Confusion matrix:\n[232  21]\n[ 51 208]\n\nAccuracy: 85.94%\nPrecision: 90.83%\nRecall: 80.31%\nF1 Score: 85.25%",
        "xgb_model_3": "Confusion matrix:\n[253   0]\n[  1 258]\n\nAccuracy: 99.80%\nPrecision: 100.00%\nRecall: 99.61%\nF1 Score: 99.81%",
        "svm_model_3": "Confusion matrix:\n[251   2]\n[  4 255]\n\nAccuracy: 98.83%\nPrecision: 99.22%\nRecall: 98.46%\nF1 Score: 98.84%",
        "xgb_model_4": "Confusion matrix:\n[241  12]\n[  5 254]\n\nAccuracy: 96.68%\nPrecision: 95.49%\nRecall: 98.07%\nF1 Score: 96.76%",
        "svm_model_4": "Confusion matrix:\n[246   7]\n[  6 253]\n\nAccuracy: 97.46%\nPrecision: 97.31%\nRecall: 97.68%\nF1 Score: 97.50%",
        "xgb_model_5": "Confusion matrix:\n[228  25]\n[ 17 242]\n\nAccuracy: 91.80%\nPrecision: 90.64%\nRecall: 93.44%\nF1 Score: 92.02%",
        "svm_model_5": "Confusion matrix:\n[216  37]\n[ 19 240]\n\nAccuracy: 89.06%\nPrecision: 86.64%\nRecall: 92.66%\nF1 Score: 89.55%",
        "bert_classifier": "Loss: 29.73%\nAccuracy: 88.67%\nPrecision: 87.08%\nRecall: 91.12%",
    },
}


# Cleans a given string by removing URLs, HTML elements, punctuations, stop words, and extra white spaces.
def clean_text(
    string: str,
    punctuations=r"""!()-[]{};:'"\,<>./?@#$%^&*_~""",
) -> str:
    """
    A method to clean text
    """
    # Cleaning the urls
    string = re.sub(r"https?://\S+|www\.\S+", "", string)
    # Cleaning the html elements
    string = re.sub(r"<.*?>", "", string)
    # Removing the punctuations
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    # Converting the text to lower
    string = string.lower()
    # Cleaning the whitespaces
    string = re.sub(r"\s+", " ", string).strip()
    return string


# Calculates and returns the average number of words per text in a given dataset.
def avg_words(dataset):
    words_in_total = 0
    texts = 0
    for text in dataset.iterrows():
        for word in text[1][0].split(" "):
            words_in_total += 1
        texts += 1
    try:
        return round(words_in_total / texts, 2)
    except ZeroDivisionError:
        return 0.0


# Calculates and returns the average number of characters per text in a given dataset.
def avg_chars(dataset):
    chars_in_total = 0
    texts = 0
    for text in dataset.iterrows():
        for word in text[1][0].split(" "):
            for char in word:
                chars_in_total += 1
        texts += 1
    try:
        return round(chars_in_total / texts, 2)
    except ZeroDivisionError:
        return 0.0


def insert_dataset1():
    dataset = pd.read_csv(
        "datasets/dataset-mixed-classifications.csv", encoding="ISO-8859-1"
    )
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
    dataset["Title"] = dataset["Text_"]
    dataset.rename(columns={"Title": "Text"}, inplace=True)
    dataset.drop("Text_", axis=1, inplace=True)
    dataset.drop("Body", axis=1, inplace=True)
    # Text preprocessing
    for i in range(dataset.Text.shape[0]):
        dataset.loc[i, "Text"] = clean_text(dataset.loc[i, "Text"])
    dataset = dataset.dropna(subset="Classification").loc[dataset.Classification != 3]
    dataset_pv_augmentation = pd.read_csv(
        "datasets/data_augmentation_privacy_violations.csv", encoding="ISO-8859-1"
    )
    dataset_pv_augmentation.drop("Unnamed: 0", axis=1, inplace=True)
    # Text preprocessing
    for i in range(dataset_pv_augmentation.Text.shape[0]):
        dataset_pv_augmentation.loc[i, "Text"] = clean_text(
            dataset_pv_augmentation.loc[i, "Text"]
        )
    # Concat & shuffle
    dataset = pd.concat([dataset, dataset_pv_augmentation])
    dataset = dataset.sample(frac=1)
    client.PVC.datasets.insert_one(
        {"SCE_dataset": dataset.to_dict("list"), "dataset_number": 1}
    )



def insert_dataset2():
    dataset = pd.read_csv("datasets/jenny-dataset.csv", encoding="ISO-8859-1")
    # Ignoring unnecessary columns
    dataset.drop("ï»¿Dataset ID", axis=1, inplace=True)
    dataset.drop("Issue ID", axis=1, inplace=True)
    dataset.drop("Source", axis=1, inplace=True)
    # Concat the Issue Summary column to the Issue Description column into a new column - Text
    dataset["Text"] = (
        dataset["Issue Summary"].astype(str)
        + " "
        + dataset["Issue Description"].astype(str)
    )
    # The only columns we are interested in are: 'Classification' & 'Text'
    dataset.drop("Issue Summary", axis=1, inplace=True)
    dataset.drop("Issue Description", axis=1, inplace=True)
    # Text preprocessing
    for i in range(dataset.Text.shape[0]):
        dataset.loc[i, "Text"] = clean_text(dataset.loc[i, "Text"])
    # Renaming and shuffling
    dataset.rename(columns={"Label": "Classification_"}, inplace=True)
    dataset.dropna(subset="Classification_", inplace=True)
    dataset["Title"] = dataset["Classification_"]
    dataset.rename(columns={"Title": "Classification"}, inplace=True)
    dataset.drop("Classification_", axis=1, inplace=True)
    dataset = dataset.sample(frac=1)
    client.PVC.datasets.insert_one(
        {"Haifa_dataset": dataset.to_dict("list"), "dataset_number": 2}
    )


def find_dataset(number, key):
    return pd.DataFrame(client.PVC.datasets.find_one({"dataset_number": number})[key])


def load_model(id, name):
    if id not in [1, 2]:
        return None, None
    try:
        if id == 1:
            if name == "bert_classifier":
                return (
                    tf.keras.models.load_model(f"models/tomer_and_eli/{name}"),
                    metrics[f"dataset{id}"][name],
                )
            else:
                return (
                    pickle.load(open(f"models/tomer_and_eli/{name}", "rb")),
                    metrics[f"dataset{id}"][name.split(".")[0]],
                )
        if id == 2:
            if name == "bert_classifier":
                return (
                    tf.keras.models.load_model(f"models/jenny/{name}"),
                    metrics[f"dataset{id}"][name],
                )
            else:
                return (
                    pickle.load(open(f"models/jenny/{name}", "rb")),
                    metrics[f"dataset{id}"][name.split(".")[0]],
                )
    except:
        return None, None


def load_feature_names(id, option=False):
    if id not in [1, 2] or type(option) != bool:
        return None
    if id == 1 and not option:
        return np.load(
            f"feature_names/tomer_and_eli/feature_names.npy", allow_pickle=True
        )
    if id == 2 and not option:
        return np.load(f"feature_names/jenny/feature_names.npy", allow_pickle=True)
    if id == 1 and option:
        return np.load(
            f"feature_names/tomer_and_eli/feature_names_25.npy", allow_pickle=True
        )
    if id == 2 and option:
        return np.load(f"feature_names/jenny/feature_names_25.npy", allow_pickle=True)


def to_tfidf(text_to_process, feature_names):
    if type(text_to_process) is str:
        text_to_process = pd.Series(text_to_process)
    if type(text_to_process) != pd.Series:
        return None
    # Create an array of zeros
    data = np.zeros((1, len(feature_names)))
    text = pd.DataFrame(data, index=[text_to_process], columns=feature_names)

    # TF-IDF preprocessing
    new_tfidf = text_to_process
    # Initialize a TfidfVectorizer object
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(new_tfidf.values.astype("U"))
    new_tfidf = pd.DataFrame(
        vector.toarray(),
        index=new_tfidf.tolist(),
        columns=vectorizer.get_feature_names_out(),
    )
    return new_tfidf, text


# Prepering the text for model prediction (TF-IDF based models)
def preprocess_text_for_predict(text_to_process, feature_names):
    if type(text_to_process) != pd.Series and type(text_to_process) != str:
        return None
    new_tfidf, text = to_tfidf(text_to_process, feature_names)
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


def init_ft():
    model_en = fasttext.load_model("English-Dict-Ft/cc.en.300.bin")
    return model_en


def init_BERT():
    # Keras layers
    bert_preprocess = hub.KerasLayer(
        hub.load("Keras-Layers/bert_en_uncased_preprocess_3")
    )
    bert_encoder = hub.KerasLayer(
        hub.load("Keras-Layers/bert_en_uncased_L-12_H-768_A-12_4")
    )
    return bert_preprocess, bert_encoder


def get_text_embedding(text):
    if type(text) != pd.Series and type(text) != str:
        return None
    bert_preprocess, bert_encoder = init_BERT()
    preprocessed_text = bert_preprocess(text)
    return bert_encoder(preprocessed_text)["pooled_output"]


def preprocess_for_ft(vectors, df):
    model_en = init_ft()
    common_words = []
    words_vecs = []
    # Saving all the common words
    for col in df.columns.tolist():
        if model_en.get_word_id(col) != -1:
            common_words.append(col)
            words_vecs.append(model_en.get_word_vector(col))
    # Getting word vectors for common words
    for issue, columns in df.iterrows():
        isuue_vector = np.zeros((300,), dtype="float32")
        counter = 0
        for i in range(len(df.columns.tolist())):
            if columns.tolist()[i] > 0 and df.columns.tolist()[i] in common_words:
                index = common_words.index(df.columns.tolist()[i])
                isuue_vector += words_vecs[index]
                counter += 1
        if counter > 0:
            vectors.append(isuue_vector / counter)
        else:
            vectors.append(isuue_vector)
    return vectors


def new_prediction(model, num, dataset, text_to_predict):
    options = [0, 1, 2]
    feature_names = []
    if dataset == 1:
        feature_names = load_feature_names(1)
    if dataset == 2:
        feature_names = load_feature_names(2)
    if dataset != 1 and dataset != 2:
        return None

    if num in options:
        if num == 1:
            if dataset == 1:
                feature_names = load_feature_names(1, True)
            if dataset == 2:
                feature_names = load_feature_names(2, True)
        pred = preprocess_text_for_predict(text_to_predict, feature_names)

    if num == 3:
        pred, _ = to_tfidf(text_to_predict, feature_names)
        pred = preprocess_for_ft([], pred)

    if num == 4:
        pred = get_text_embedding((pd.Series(text_to_predict)))

    if num == 5:
        pred = model.predict(pd.Series(text_to_predict))[0]
        # Result between 0.0 to 1.0
        if pred > 0.5:
            return 1
        else:
            return 0

    if num > 5 or num < 0:
        return None

    return model.predict(pred)[0]
