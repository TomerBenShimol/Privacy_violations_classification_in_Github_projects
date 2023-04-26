# import numpy as np
# import pickle as pk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV

# # def load_model(name):
# #     path = f"/Users/tomerbenshimol/Desktop/Software_Engineering_Project/models/jenny/{name}.pickle"
# #     return pickle.load(open(filename, "rb"))



# # Prepering the text for model prediction
# def preprocess_text_for_predict(text_to_process, feature_names):
#     if type(text_to_process) is str:
#         text_to_process = pd.Series(text_to_process)
#     # Create an array of zeros
#     data = np.zeros((1, len(feature_names)))
#     text = pd.DataFrame(data,index=[text_to_process], columns=feature_names)

#     # TF-IDF preprocessing
#     new_tfidf = text_to_process
#     # Initialize a TfidfVectorizer object
#     vectorizer = TfidfVectorizer()
#     vector = vectorizer.fit_transform(new_tfidf.values.astype('U'))
#     new_tfidf = pd.DataFrame(vector.toarray(), index=new_tfidf.tolist(), columns=vectorizer.get_feature_names_out())
#     weights = []

#     # Saving tfidf original values
#     for i in range(new_tfidf.shape[0]):
#         for j in range(new_tfidf.shape[1]):
#             weights.append(new_tfidf.iloc[i][j])

#     # new_tfidf feature names
#     columns = new_tfidf.columns.tolist()

#     # If they have common feature vector the tfidf value changes to new_tfidf value, else it remains zero
#     for i in range(len(columns)):
#         if columns[i] in text.columns:
#             index = list(text.columns).index(columns[i])
#             text.iloc[0][index] = weights[i]
#     return text