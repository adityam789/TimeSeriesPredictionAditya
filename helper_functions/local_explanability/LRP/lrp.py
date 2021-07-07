import configparser as cp
import pickle
import matplotlib.pyplot as plt
from lstm_network import LSTM_network
import tensorflow as tf
import numpy as np

from keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'train_x'), 'rb') as f:
	X_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	train_y = pickle.load( f)

# create the lstm-lrp model
n_hidden = 50
embedding_dim = 1
n_classes = 1
weights = model.get_weights()
for i in weights:
	print(i.shape)

# our keras model has no bias in the final dense layer. Therefore we add a bias of zero to the weights
weights.append(np.zeros((n_classes,)))
lrp_model = LSTM_network(n_hidden, embedding_dim, n_classes, weights=weights)

# explain the classification
eps = 1e-3
bias_factor = 0.0
# by setting y=None, the relevances will be calculated for the predicted class of the sample. We recommend this
# usage, however, if you are interested in the relevances towards the 1st class, you could use y = np.array([1])
explanation, Rest = lrp_model.lrp(X_train, eps=eps, bias_factor=bias_factor)
print(explanation.shape)

revelances = tf.reduce_sum(explanation, axis=2)
print(revelances)