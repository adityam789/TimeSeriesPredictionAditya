import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import shap
import pickle
import configparser as cp

from keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'train_x'), 'rb') as f:
	X_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	X_test=pickle.load( f)

regressor = model
pred_x = np.argmax(model.predict(X_train), axis=-1)
random_ind = np.random.choice(X_train.shape[0], 1000, replace=False)
# print(random_ind)
data = X_train[random_ind[0:500]]
e = shap.DeepExplainer((regressor.layers[0].input, regressor.layers[-1].output),data)
test1 = X_train[random_ind[500:1000]]
shap_val = e.shap_values(test1)
shap_val = np.array(shap_val)
shap_val = np.reshape(shap_val,(int(shap_val.shape[1]),int(shap_val.shape[2]),int(shap_val.shape[3])))
shap_abs = np.absolute(shap_val)
sum_0 = np.sum(shap_abs,axis=0)
f_names = ['stock']
x_pos = [i for i, _ in enumerate(f_names)]
plt1 = plt.subplot(311)
plt1.barh(x_pos,sum_0[1])
plt1.set_yticks(x_pos)
plt1.set_yticklabels(f_names)
plt1.set_title("Yesterday’s features (time-step 2)")
plt2 = plt.subplot(312,sharex=plt1)
plt2.barh(x_pos,sum_0[0])
plt2.set_yticks(x_pos)
plt2.set_yticklabels(f_names)
plt2.set_title('The day before yesterday’s features(time-step 1)')
plt.tight_layout()
plt.show()