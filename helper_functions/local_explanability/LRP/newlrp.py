from code.LSTM.LSTM_bidi import * 
from code.util.heatmap import html_heatmap
import configparser as cp
import pickle
import numpy as np  
from keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'train_x'), 'rb') as f:
	X_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	train_y = pickle.load( f)

def predict(words):
    """Returns the classifier's predicted class"""
    net                 = LSTM_bidi()                                   # load trained LSTM model
    w_indices           = [net.voc.index(w) for w in words]             # convert input sentence to word IDs
    net.set_input(w_indices)                                            # set LSTM input sequence
    scores              = net.forward()                                 # classification prediction scores
    return np.argmax(scores)