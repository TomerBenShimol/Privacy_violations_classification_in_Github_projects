import unittest
from functions import *
import tensorflow as tf
import pandas as pd
import fasttext


class CleanTextTestCase(unittest.TestCase):
    def test_clean_text_with_url(self):
        string = "Check out this website: https://www.example.com"
        expected_output = "check out this website"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_html_elements(self):
        string = "This is <b>bold</b> and <i>italic</i>"
        expected_output = "this bold italic"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_punctuations(self):
        string = "Hello! How are you?"
        expected_output = "hello how are you"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_stop_words(self):
        string = "This is the example to reproduce"
        expected_output = "this example"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_extra_white_spaces(self):
        string = "    Remove     extra     spaces   "
        expected_output = "remove extra spaces"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_combination(self):
        string = "Check out this website: https://www.example.com. It has <b>bold</b> text and some punctuations!"
        expected_output = "check out this website it has bold text some punctuations"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_empty_string(self):
        string = ""
        expected_output = ""
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_only_url(self):
        string = "https://www.example.com"
        expected_output = ""
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_only_html_elements(self):
        string = "<b>bold</b> <i>italic</i>"
        expected_output = "bold italic"
        self.assertEqual(clean_text(string), expected_output)

    def test_clean_text_with_only_punctuations(self):
        string = "!@#$%"
        expected_output = ""
        self.assertEqual(clean_text(string), expected_output)


class AvgWordsTestCase(unittest.TestCase):
    def test_avg_words_with_empty_dataset(self):
        dataset = pd.DataFrame([])
        expected_output = 0.0
        self.assertEqual(avg_words(dataset), expected_output)

    def test_avg_words_with_single_text(self):
        dataset = pd.DataFrame([("Hello world")])
        expected_output = 2.0
        self.assertEqual(avg_words(dataset), expected_output)

    def test_avg_words_with_multiple_texts(self):
        dataset = pd.DataFrame([("Hello world"), ("This is a sentence")])
        expected_output = 3.0
        self.assertEqual(avg_words(dataset), expected_output)

    def test_avg_words_with_texts_containing_no_spaces(self):
        dataset = pd.DataFrame([("Hello"), ("This")])
        expected_output = 1.0
        self.assertEqual(avg_words(dataset), expected_output)


class AvgCharsTestCase(unittest.TestCase):
    def test_avg_chars_with_empty_dataset(self):
        dataset = pd.DataFrame([[""]])
        expected_output = 0.0
        self.assertEqual(avg_chars(dataset), expected_output)

    def test_avg_chars_with_single_text(self):
        dataset = pd.DataFrame([["Hello world"]])
        expected_output = 10.0
        self.assertEqual(avg_chars(dataset), expected_output)

    def test_avg_chars_with_multiple_texts(self):
        dataset = pd.DataFrame([["Hello world"], ["This is a sentence"]])
        expected_output = 12.5
        self.assertEqual(avg_chars(dataset), expected_output)

    def test_avg_chars_with_texts_containing_no_characters(self):
        dataset = pd.DataFrame([[""], [""]])
        expected_output = 0.0
        self.assertEqual(avg_chars(dataset), expected_output)


class ReadDataset1TestCase(unittest.TestCase):
    def test_read_dataset1_returns_dataframe(self):
        dataset = read_dataset1()
        self.assertIsInstance(dataset, pd.DataFrame)

    def test_read_dataset1_has_expected_columns(self):
        dataset = read_dataset1()
        expected_columns = ["Text", "Classification"]
        self.assertListEqual(list(dataset.columns), expected_columns)

    def test_read_dataset1_text_preprocessing(self):
        dataset = read_dataset1()
        self.assertTrue(all(dataset["Text"].apply(lambda x: isinstance(x, str))))

    def test_read_dataset1_classification_values(self):
        dataset = read_dataset1()
        valid_classifications = [0, 1, 3]
        self.assertTrue(all(dataset["Classification"].isin(valid_classifications)))


class ReadDataset2TestCase(unittest.TestCase):
    def test_read_dataset2_returns_dataframe(self):
        dataset = read_dataset2()
        self.assertIsInstance(dataset, pd.DataFrame)

    def test_read_dataset2_has_expected_columns(self):
        dataset = read_dataset2()
        expected_columns = ["Text", "Classification"]
        self.assertListEqual(list(dataset.columns), expected_columns)

    def test_read_dataset2_text_preprocessing(self):
        dataset = read_dataset2()
        self.assertTrue(all(dataset["Text"].apply(lambda x: isinstance(x, str))))

    def test_read_dataset2_classification_values(self):
        dataset = read_dataset2()
        valid_classifications = [0, 1]
        self.assertTrue(all(dataset["Classification"].isin(valid_classifications)))


class LoadModelTestCase(unittest.TestCase):
    def test_load_model_with_invalid_id(self):
        id = 3
        name = "bert_classifier"
        expected_output = (None, None)
        self.assertEqual(load_model(id, name), expected_output)

    def test_load_model_with_invalid_name(self):
        id = 1
        name = "invalid_model.pkl"
        expected_output = (None, None)
        self.assertEqual(load_model(id, name), expected_output)

    def test_load_model_with_exception(self):
        id = 1
        name = "bert_classifier"
        expected_output = (None, None)
        self.assertNotEqual(load_model(id, name), expected_output)

    def test_load_model_not_none(self):
        model, metric = load_model(1, "xgb_model_1.pickle")
        self.assertIsNotNone(model)
        self.assertIsNotNone(metric)

    def test_load_model_wrong_name(self):
        model, metric = load_model(1, "model that does not exists")
        self.assertIsNone(model)
        self.assertIsNone(metric)

    def test_load_model_wrong_id(self):
        model, metric = load_model(4, "xgb_model_1.pickle")
        self.assertIsNone(model)
        self.assertIsNone(metric)


class LoadFeatureNamesTestCase(unittest.TestCase):
    def test_load_feature_names_with_id_1_and_option_false(self):
        feature_names = load_feature_names(1, option=False)
        self.assertIsInstance(feature_names, np.ndarray)

    def test_load_feature_names_with_id_2_and_option_false(self):
        feature_names = load_feature_names(2, option=False)
        self.assertIsInstance(feature_names, np.ndarray)

    def test_load_feature_names_with_id_1_and_option_true(self):
        feature_names = load_feature_names(1, option=True)
        self.assertIsInstance(feature_names, np.ndarray)

    def test_load_feature_names_with_id_2_and_option_true(self):
        feature_names = load_feature_names(2, option=True)
        self.assertIsInstance(feature_names, np.ndarray)

    def test_load_feature_names_with_invalid_id(self):
        feature_names = load_feature_names(3, option=False)
        self.assertIsNone(feature_names)

    def test_load_feature_names_with_invalid_option(self):
        feature_names = load_feature_names(1, option="True")
        self.assertIsNone(feature_names)


class ToTfidfTestCase(unittest.TestCase):
    def test_to_tfidf_with_string_input(self):
        text_to_process = "This is a sample text"
        feature_names = ["sample", "text"]
        new_tfidf, text = to_tfidf(text_to_process, feature_names)
        self.assertIsInstance(new_tfidf, pd.DataFrame)
        self.assertIsInstance(text, pd.DataFrame)

    def test_to_tfidf_with_series_input(self):
        text_to_process = pd.Series(["This is a sample text"])
        feature_names = ["sample", "text"]
        new_tfidf, text = to_tfidf(text_to_process, feature_names)
        self.assertIsInstance(new_tfidf, pd.DataFrame)
        self.assertIsInstance(text, pd.DataFrame)

    def test_to_tfidf_with_invalid_input(self):
        text_to_process = 12345  # Invalid input type
        feature_names = ["sample", "text"]
        result = to_tfidf(text_to_process, feature_names)
        self.assertIsNone(result)


class PreprocessTextForPredictTestCase(unittest.TestCase):
    def test_preprocess_text_for_predict_with_valid_input(self):
        text_to_process = "This is a sample text"
        feature_names = ["sample", "text"]
        result = preprocess_text_for_predict(text_to_process, feature_names)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, len(feature_names)))

    def test_preprocess_text_for_predict_with_empty_feature_names(self):
        text_to_process = "This is a sample text"
        feature_names = []
        result = preprocess_text_for_predict(text_to_process, feature_names)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (1, 0))

    def test_preprocess_text_for_predict_with_invalid_input(self):
        text_to_process = 12345  # Invalid input type
        feature_names = ["sample", "text"]
        result = preprocess_text_for_predict(text_to_process, feature_names)
        self.assertIsNone(result)


class InitFtTestCase(unittest.TestCase):
    def test_init_ft_returns_model(self):
        model = init_ft()
        self.assertIsNotNone(model)


class PreprocessForFtTestCase(unittest.TestCase):
    def setUp(self):
        # Set up a sample dataframe
        self.df = pd.DataFrame(
            {
                "word1": [1, 0, 1],
                "word2": [0, 1, 0],
                "word3": [1, 1, 0],
            }
        )

    def tearDown(self):
        # Clean up any resources used in the tests
        pass

    def test_preprocess_for_ft_with_valid_input(self):
        vectors = []
        model_en = init_ft()
        result = preprocess_for_ft(vectors, self.df)
        self.assertEqual(len(result), len(self.df))
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (300,))

    def test_preprocess_for_ft_with_empty_dataframe(self):
        empty_df = pd.DataFrame()
        vectors = []
        model_en = init_ft()
        result = preprocess_for_ft(vectors, empty_df)
        self.assertEqual(len(result), 0)

    def test_preprocess_for_ft_with_no_common_words(self):
        df_no_common_words = pd.DataFrame(
            {
                "word4": [1, 0, 1],
                "word5": [0, 1, 0],
                "word6": [1, 1, 0],
            }
        )
        vectors = []
        model_en = init_ft()
        result = preprocess_for_ft(vectors, df_no_common_words)
        self.assertEqual(len(result), len(df_no_common_words))
        self.assertIsInstance(result[0], np.ndarray)
        self.assertEqual(result[0].shape, (300,))


class NewPredictionTestCase(unittest.TestCase):
    def setUp(self):
        # Set up a sample dataset
        self.dataset = 1
        # Set up a sample model
        self.model = load_model(1, "xgb_model_1.pickle")
        # Set up a sample text to predict
        self.text_to_predict = "Sample text"

    def tearDown(self):
        # Clean up any resources used in the tests
        pass

    def test_new_prediction_with_option_1_dataset_1(self):
        result = new_prediction(self.model, 1, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_1_dataset_2(self):
        result = new_prediction(self.model, 1, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_2_dataset_1(self):
        result = new_prediction(self.model, 2, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_2_dataset_2(self):
        result = new_prediction(self.model, 2, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_3(self):
        result = new_prediction(self.model, 3, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_4(self):
        result = new_prediction(self.model, 4, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_option_5(self):
        result = new_prediction(self.model, 5, self.dataset, self.text_to_predict)
        self.assertIsNotNone(result)

    def test_new_prediction_with_invalid_option(self):
        result = new_prediction(self.model, 6, self.dataset, self.text_to_predict)
        self.assertIsNone(result)


#     def test_get_text_embedding_with_invalid_text_type(self):
#         text = 12345  # Invalid input type
#         result = get_text_embedding(text)
#         self.assertIsNone(result)


# class TestLoadFeatureNames(unittest.TestCase):
#     def test_load_feature_names(self):
#         feature_names = load_feature_names(1)
#         self.assertIsNotNone(feature_names)


# class TestToTfidf(unittest.TestCase):
#     def test_to_tfidf(self):
#         text_to_process = "example text"
#         feature_names = ["example", "text"]
#         output, _ = to_tfidf(text_to_process, feature_names)
#         self.assertIsNotNone(output)


# class TestPreprocessTextForPredict(unittest.TestCase):
#     def test_preprocess_text_for_predict(self):
#         text_to_process = "example text"
#         feature_names = ["example", "text"]
#         output = preprocess_text_for_predict(text_to_process, feature_names)
#         self.assertIsNotNone(output)


# class TestInitFt(unittest.TestCase):
#     def test_init_ft(self):
#         model_en = init_ft()
#         self.assertIsNotNone(model_en)


# class TestInitBERT(unittest.TestCase):
#     def test_init_BERT(self):
#         bert_preprocess, bert_encoder = init_BERT()
#         self.assertIsNotNone(bert_preprocess)
#         self.assertIsNotNone(bert_encoder)


# class TestGetTextEmbedding(unittest.TestCase):
#     def test_get_text_embedding(self):
#         text = ["example text"]
#         embedding = get_text_embedding(text)
#         self.assertIsNotNone(embedding)


# class TestPreprocessForFt(unittest.TestCase):
#     def test_preprocess_for_ft(self):
#         vectors = []
#         df = pd.DataFrame({"example": [1], "text": [1]})
#         output = preprocess_for_ft(vectors, df)
#         self.assertIsNotNone(output)


# class TestNewPrediction(unittest.TestCase):
#     def test_new_prediction(self):
#         # Note: You need to have a valid model saved in the respective folder to test this function.
#         model = pickle.load(open("models/your_model_folder/your_model_name_here", "rb"))
#         num = 0
#         dataset = 1
#         text_to_predict = "example text"
#         prediction = new_prediction(model, num, dataset, text_to_predict)
#         self.assertIsNotNone(prediction)


if __name__ == "__main__":
    unittest.main()
