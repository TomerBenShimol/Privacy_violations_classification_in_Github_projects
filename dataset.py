import pandas as pd

def read_dataset():
	dataset = pd.read_csv("../datasets/dataset-mixed-classifications.csv", encoding = "ISO-8859-1")

	# Ignoring unnecessary columns 
	dataset.drop("ï»¿URL", axis=1, inplace=True)
	dataset.drop("ID", axis=1, inplace=True)
	dataset.drop("Classification - Tomer", axis=1, inplace=True)
	dataset.drop("Classification - Eli", axis=1, inplace=True)
	dataset.drop("Disagreements - Red", axis=1, inplace=True)
	dataset.dropna(subset="Classification", inplace=True)
	dataset = dataset.sample(frac=1)
	return dataset
