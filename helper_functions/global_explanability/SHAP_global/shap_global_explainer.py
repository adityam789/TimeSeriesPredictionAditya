import configparser as cp
import pickle
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import numpy as np
import shap

model = load_model('Results\model_paths\stock_lstm_model.h5')

config = cp.RawConfigParser()
config.read('config/config.properties')
with open(config.get('data_path', 'train_x'), 'rb') as f:
    X_train = pickle.load(f)
with open(config.get('data_path', 'test_x'), 'rb') as f:
    y_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
    X_test=pickle.load( f)

def f(x):
    x_2 = x.reshape(x.shape[0], x.shape[1], 1)
    return model(x_2)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

explainer = shap.KernelExplainer(f, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=1)

shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit", show=False).savefig(config.get('vis','vis_path_folder8') + 'shap_global_explain')