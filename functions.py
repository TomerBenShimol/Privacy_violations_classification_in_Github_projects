import pandas as pd
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import fasttext
import tensorflow_hub as hub
import tensorflow_text
from sklearn.feature_extraction.text import TfidfVectorizer

metrics = {
    "dataset1": {
        "xgb_model_1": "Confusion matrix:\n[97 18]\n[22 60]\n\nAccuracy: 79.70%\nPrecision: 76.92%\nRecall: 73.17%\nF1 Score: 75.00%",
        "svm_model_1": "Confusion matrix:\n[106  9]\n[15   67]\n\nAccuracy: 87.82%\nPrecision: 88.16%\nRecall: 81.71%\nF1 Score: 84.81%",
        "xgb_model_2": "Confusion matrix:\n[97 18]\n[27 55]\n\nAccuracy: 77.16%\nPrecision: 75.34%\nRecall: 67.07%\nF1 Score: 70.97%",
        "svm_model_2": "Confusion matrix:\n[95 20]\n[39 43]\n\nAccuracy: 70.05%\nPrecision: 68.25%\nRecall: 52.44%\nF1 Score: 59.31%",
        "xgb_model_3": "Confusion matrix:\n[92 23]\n[20 62]\n\nAccuracy: 78.17%\nPrecision: 72.94%\nRecall: 75.61%\nF1 Score: 74.25%",
        "svm_model_3": "Confusion matrix:\n[103  12]\n[15   67]\n\nAccuracy: 86.29%\nPrecision: 84.81%\nRecall: 81.71%\nF1 Score: 83.23%",
        "xgb_model_4": "Confusion matrix:\n[98 17]\n[22 60]\n\nAccuracy: 80.20%\nPrecision: 77.92%\nRecall: 73.17%\nF1 Score: 75.47%",
        "svm_model_4": "Confusion matrix:\n[88 27]\n[20 62]\n\nAccuracy: 76.14%\nPrecision: 69.66%\nRecall: 75.61%\nF1 Score: 72.51%",
        "xgb_model_5": "Confusion matrix:\n[83 32]\n[28 54]\n\nAccuracy: 69.54%\nPrecision: 62.79%\nRecall: 65.85%\nF1 Score: 64.29%",
        "svm_model_5": "Confusion matrix:\n[99  16]\n[53 29]\n\nAccuracy: 64.97%\nPrecision: 64.44%\nRecall: 35.37%\nF1 Score: 45.67%",
        "bert_classifier": "Loss: 63.22%\nAccuracy: 60.41%\n\nPrecision: 53.33%\nRecall: 39.02%",
    },
    "dataset2": {
        "xgb_model_1": "Confusion matrix:\n[278  66]\n[ 15 408]\n\nAccuracy: 89.44%\nPrecision: 86.08%\nRecall: 96.45%\nF1 Score: 90.97%",
        "svm_model_1": "Confusion matrix:\n[318  26]\n[  6 417]\n\nAccuracy: 95.83%\nPrecision: 94.13%\nRecall: 98.58%\nF1 Score: 96.30%",
        "xgb_model_2": "Confusion matrix:\n[324  20]\n[138 285]\n\nAccuracy: 79.40%\nPrecision: 93.44%\nRecall: 67.38%\nF1 Score: 78.30%",
        "svm_model_2": "Confusion matrix:\n[320  24]\n[132 291]\n\nAccuracy: 79.66%\nPrecision: 92.38%\nRecall: 68.79%\nF1 Score: 78.86%",
        "xgb_model_3": "Confusion matrix:\n[280  64]\n[ 10 413]\n\nAccuracy: 90.35%\nPrecision: 86.58%\nRecall: 97.64%\nF1 Score: 91.78%",
        "svm_model_3": "Confusion matrix:\n[325  19]\n[  9 414]\n\nAccuracy: 96.35%\nPrecision: 95.61%\nRecall: 97.87%\nF1 Score: 96.73%",
        "xgb_model_4": "Confusion matrix:\n[308  36]\n[ 16 407]\n\nAccuracy: 93.22%\nPrecision: 91.87%\nRecall: 96.22%\nF1 Score: 94.00%",
        "svm_model_4": "Confusion matrix:\n[320  24]\n[ 15 408]\n\nAccuracy: 94.92%\nPrecision: 94.44%\nRecall: 96.45%\nF1 Score: 95.44%",
        "xgb_model_5": "Confusion matrix:\n[311  33]\n[ 18 405]\n\nAccuracy: 93.35%\nPrecision: 92.47%\nRecall: 95.74%\nF1 Score: 94.08%",
        "svm_model_5": "Confusion matrix:\n[305  39]\n[ 19 404]\n\nAccuracy: 92.44%\nPrecision: 91.20%\nRecall: 95.51%\nF1 Score: 93.30%",
        "bert_classifier": "Loss: 30.79%\nAccuracy: 88.14%\nPrecision: 85.78%\nRecall: 94.09%",
    },
}


# Cleans a given string by removing URLs, HTML elements, punctuations, stop words, and extra white spaces.
def clean_text(
    string: str,
    punctuations_with_whitespace=r"""!()-[]{};:",<>./?@#$%^&*_~""",
    punctuations_without_whitespace=r"""\'/""",
    stop_words=[
        "a",
        "an",
        "the",
        "this",
        "that",
        "is",
        "it",
        "to",
        "and",
        "be",
        "will",
    ],
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
        if x in punctuations_with_whitespace:
            string = string.replace(x, " ")
        if x in punctuations_without_whitespace:
            string = string.replace(x, "")

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = " ".join([word for word in string.split() if word not in stop_words])

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


def read_dataset1():
    dataset = pd.read_csv("datasets/SCE_dataset.csv", encoding="ISO-8859-1")
    return dataset.sample(frac=1)


def read_dataset2():
    dataset = pd.read_csv("datasets/Haifa_dataset.csv", encoding="ISO-8859-1")
    dataset.dropna(subset="Text", inplace=True)
    return dataset.sample(frac=1)


def load_model(id, name):
    if id not in [1, 2]:
        return None, None
    try:
        if id == 1:
            if name == "bert_classifier":
                return (
                    tf.keras.models.load_model(f"models/SCE/{name}"),
                    metrics[f"dataset{id}"][name],
                )
            else:
                return (
                    pickle.load(open(f"models/SCE/{name}", "rb")),
                    metrics[f"dataset{id}"][name.split(".")[0]],
                )
        if id == 2:
            if name == "bert_classifier":
                return (
                    tf.keras.models.load_model(f"models/Haifa/{name}"),
                    metrics[f"dataset{id}"][name],
                )
            else:
                return (
                    pickle.load(open(f"models/Haifa/{name}", "rb")),
                    metrics[f"dataset{id}"][name.split(".")[0]],
                )
    except:
        return None, None


def load_feature_names(id, option=False):
    if id not in [1, 2] or type(option) != bool:
        return None
    if id == 1 and not option:
        return np.load(f"feature_names/SCE/feature_names.npy", allow_pickle=True)
    if id == 2 and not option:
        return np.load(f"feature_names/Haifa/feature_names.npy", allow_pickle=True)
    if id == 1 and option:
        return np.load(f"feature_names/SCE/feature_names_25.npy", allow_pickle=True)
    if id == 2 and option:
        return np.load(f"feature_names/Haifa/feature_names_25.npy", allow_pickle=True)


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
