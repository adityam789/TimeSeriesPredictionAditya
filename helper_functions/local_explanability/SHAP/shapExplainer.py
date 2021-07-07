import configparser as cp
import pickle

import shap

# import tensorflow.compat.v1.keras.backend as K
# from K.models import load_model
from tensorflow.keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

from keras.models import load_model
model2 = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open( config.get('data_path', 'train_x'), 'rb') as f:
	X_train=pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	X_test=pickle.load( f)

print(model2.layers[-1].output)

explainer = shap.DeepExplainer(model, X_train)
# explain the the testing instances (can use fewer instanaces)
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(X_test)
# init the JS visualization code
print(shap_values)
print('HomelessMan')
# shap.initjs()
# shap.force_plot(explainer.expected_value[100], shap_values[100][100])

