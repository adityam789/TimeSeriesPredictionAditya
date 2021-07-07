import configparser as cp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

from keras.models import load_model
model2 = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'test_y'), 'rb') as f:
    y_test=pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	X_test=pickle.load( f)

def feature_importance(x, y, func, repeat=10, plot=True):
    base_score = func(x, y)
    imp = [0] * x.shape[1]
    for i in range(repeat):
        for c in range(x.shape[1]):
            t = x.copy()
            score = func(t, y, c)
            imp[c] += base_score - score
    imp = [a/repeat for a in imp]
    # imp_df = pd.DataFrame(data=[imp], columns=x.columns)
    # if plot:
    #     imp_df.sum().sort_values().plot.barh()
    return imp

def get_avg(x, y, c = None):
    if c != None:
        col = x[:, c]
        np.random.shuffle(col)
        modelOut = model.evaluate(x, y, verbose=0, return_dict=True)
        return modelOut['loss']
    else:
        modelOut = model.evaluate(x, y, verbose=0, return_dict=True)
        return modelOut['loss']

imp = feature_importance(X_test.copy(), y_test, get_avg, repeat=1)

plt.plot(imp)

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder3')):
    os.makedirs( config.get('vis','vis_path_folder3'))
plt.savefig(config.get('vis','vis_path_folder3') + '/xai_explain.png')

plt.show()
