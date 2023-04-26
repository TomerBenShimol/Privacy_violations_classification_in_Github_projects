import streamlit as st
from functions import * 
import pandas as pd

header = st.container()
dataset_section = st.container()
model = st.container()

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with header:
	st.markdown('<h1 class="dataset1-main-header">This dataset was annotated by</h1>', unsafe_allow_html=True)
	st.markdown('<h2 class="dataset1-secondary-header"><a href="https://www.linkedin.com/in/tomerbenshimol/" class="link1">Tomer Ben Shimol</a> & <a href="https://www.linkedin.com/in/eli-amuyev-224a4b210/" class="link1">Eliyahu Amuyev</a></h2>', unsafe_allow_html=True)

with dataset_section:
 # Import data
 dataset = read_dataset1()
 st.write(dataset.head())
 st.text(f'{dataset.shape[0]} Annotations.')
 st.text(f'{dataset.Classification.value_counts()[0]} Non-privacy violations.')
 st.text(f'{dataset.Classification.value_counts()[1]} Privacy violations.')
 st.text('* After performing data augmentation - with the help of chatGPT.')
 st.text('With the help of YAKE! we extracted keywords from the texts that were labeled as - 1')
 st.image('img/pv_d1.png')

with model:
 st.header('Model selection')
 options1 = ('Regular TF-IDF', 'Keywords x TF-IDF', 'Normaized TF-IDF', 'fastText', 'BERT', 'BERT Classifier')
 options2 =  ('eXtreme Gradient Booster', 'Support vector machine')
 option1 = st.selectbox(
    'Which model would you like to try?',
    options1)
 index1 = options1.index(option1)
 if index1 != 5:
    option2 = st.selectbox(
        'Classifier?',
        options2)
    index2 = options2.index(option2)
 else:
  index2 = None
 st.text("* Please enter text that describes for you a type of privacy violation\n  in software.")
 text = st.text_input("", placeholder="Write some text to classify...", key="23")
 model_name = 'xgb'
 if index2 and index2 == 1:
  model_name = 'svm'
 model_name = f'{model_name}_model_{index1 + 1}.pickle'
 if index1 == 5:
   model_name = 'bert_classifier.pickle'
 model = load_model(1, model_name)
 feature_names = load_feature_names(1)
 result = -1
 if text != '':
  result = new_prediction(model, text, feature_names)
 if result == 1:
  st.write('Privacy Violation! ⛔️')
 if result == 0:
  st.write('Not a Privacy Violation 🙏🏻')
#   st.write(st.session_state[23])