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


# Cleans a given string by removing URLs, HTML elements, punctuations, stop words, and extra white spaces.
def clean_text(
    string: str,
    punctuations=r"""!()-[]{};:'"\,<>./?@#$%^&*_~""",
    stop_words=["the", "a", "and", "is", "be", "will", "steps", "to", "reproduce"],
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
        for word in text[1][1].split(" "):
            words_in_total += 1
        texts += 1
    return round(words_in_total / texts, 2)


# Calculates and returns the average number of characters per text in a given dataset.
def avg_chars(dataset):
    chars_in_total = 0
    texts = 0
    for text in dataset.iterrows():
        for word in text[1][1]:
            chars_in_total += 1
        texts += 1
    return round(chars_in_total / texts, 2)


def read_dataset1():
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
    return dataset


def read_dataset2():
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
    return dataset


def load_model(id, name):
    if id == 1:
        if name == "bert_classifier":
            return tf.keras.models.load_model(f"models/tomer_and_eli/{name}")
        else:
            return pickle.load(open(f"models/tomer_and_eli/{name}", "rb"))
    if id == 2:
        if name == "bert_classifier":
            return tf.keras.models.load_model(f"models/jenny/{name}")
        else:
            return pickle.load(open(f"models/jenny/{name}", "rb"))


def load_feature_names(id, option=False):
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


# Only once
def init_ft():
    model_en = fasttext.load_model("English-Dict-Ft/cc.en.300.bin")
    return model_en


# Only once
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

    return model.predict(pred)[0]
