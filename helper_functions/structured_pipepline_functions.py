from helper_functions import preprocessing
from helper_functions import modelling as model

def pre_processing(path = None): 
	preprocessing.preprocessing(path)

def modelling():
    model.model_making()