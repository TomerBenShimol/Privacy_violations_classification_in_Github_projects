import streamlit as st

header = st.container()
approach = st.container()
goal = st.container()
datasets = st.container()

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	
with header:
  st.title('Welcome to PVC')
  st.header("The world's first privacy violation classifier!")
  st.text("Privacy violations occur when personal information is accessed or used without\nconsent and detecting them is important to prevent harm and protect privacy.\nIn this project, we trained machine-learning models that can accurately classify\nissues related to privacy violations.")

with approach:
  st.header("Approach")
  st.text("We experimented with different text representation and classification models\nlike Support vector machine, eXtreme Gradient Booster, etc. and evaluated their\naccuracy on our dataset.")
        
with goal:
  st.header('Main goal')
  st.text("Our findings can assist GitHub users in managing privacy issues more effectively.\nAn accurate classification could enhance trust and help protect sensitive\ninformation, benefitting social media and software development collaboration\nplatforms.")

with datasets:
  st.header("Datasets")
  st.text("We collected and annotated a new dataset of issues reported in GitHub projects\nthat contains 981 issues. In addition, we worked with another dataset of privacy\nissues that contains 2556 issues collected and annotated by Jenny Guber.")
  col1, col2 = st.columns(2)
  
  with col1:
   st.markdown('<div class="button1"><a href="/Dataset_1" class="link1" target="_self">Tomer & Eli</a></div>', unsafe_allow_html=True)

  with col2:
   st.markdown('<div class="button1"><a href="/Dataset_2" class="link1" target="_self">Jenny Guber</a></div>', unsafe_allow_html=True)
  st.markdown('<p>⬆️</p>', unsafe_allow_html=True)
  st.text("You have two datasets to choose from, and we have trained six different models for\neach. Select one dataset and explore the models that have been trained on it.")




# header = st.container()
# dataset = st.container()t
# approach = st.container()
# goal = st.container()
# classifier = st.container()



# with dataset:
# 	st.header("Dataset")
# 	st.text("We collected and annotated a new dataset of issues reported in GitHub projects.")
# 	# Import data 
# 	dataset = dt.read_dataset()
# 	st.write(dataset.head())

# with classifier:
# 	st.header("Try PVC")
# 	st.text("* Please enter text that describes for you a type of privacy violation\n  in software.")
# 	text = st.text_input("", placeholder="Write some text...")
# 	model = pvc.load_model("/Users/tomerbenshimol/Desktop/Software_Engineering_Project/models/jenny/xgb_model_1.pickle")
# 	feature_names = pvc.load_feature_names("/Users/tomerbenshimol/Desktop/Software_Engineering_Project/feature_names/feature_names_Jenny.npy")
# 	if text != '':
# 		result = pvc.new_prediction(model, text, feature_names)
# 		if result == 1:
# 			st.write('Privacy Violation! ⛔️')
# 		if result == 0:
# 			st.write('Not a Privacy Violation 🙏🏻')
