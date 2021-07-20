from alibi.explainers import ALE
from .plotter import plotter_ALE
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model

def ALE_explainer_function():

    model = load_model('Results\model_paths\stock_lstm_model.h5')

    config = cp.RawConfigParser()
    config.read('config/config.properties')

    with open(config.get('data_path', 'scaler_dump'), 'rb') as f:
        scaler = pickle.load(f)

    feature_names = [str(i) for i in range(100)]
    target_names = ['a']

    with open(config.get('data_path', 'train_x'), 'rb') as f:
        X_train = pickle.load(f)

    shape = X_train[-10:].shape

    ale = ALE(model.predict, feature_names=feature_names, target_names=target_names)

    exp = ale.explain(X_train[-10:].reshape(shape[0], shape[1]))

    def keyOfTuple(t):
        sum = 0
        for i in t[1]:
            sum += i[0]
        ans = sum / len(t[1])
        return ans
    arr = []
    for i, j in enumerate(exp.ale_values):
        arr.append((i,j))

    sorted_arr = sorted(arr, key=keyOfTuple)
    max10 = [str(i) for i,j in sorted_arr[-5:]]
    min10 = [str(i) for i,j in sorted_arr[:5]]

    fig = plotter_ALE(exp, max10)

    if not os.path.exists(config.get('vis','vis_path_folder6')):
        os.makedirs( config.get('vis','vis_path_folder6'))
    plt.savefig(config.get('vis','vis_path_folder6') + '/ALE_explain_max5_features.png')

    fig2 = plotter_ALE(exp, min10)

    if not os.path.exists(config.get('vis','vis_path_folder6')):
        os.makedirs( config.get('vis','vis_path_folder6'))
    plt.savefig(config.get('vis','vis_path_folder6') + '/ALE_explain_min5_features.png')

    # plt.show()


