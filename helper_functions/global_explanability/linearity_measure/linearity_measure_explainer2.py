import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import configparser as cp
import pickle
import os

from alibi.confidence.model_linearity import linearity_measure, LinearityMeasure
from alibi.confidence.model_linearity import _infer_feature_range

from tensorflow.keras import backend as K
from keras.models import load_model

def linearity_measure_explainer_function():

    model = load_model('Results\model_paths\stock_lstm_model.h5')

    config = cp.RawConfigParser()
    config.read('config/config.properties')

    with open(config.get('data_path', 'train_x'), 'rb') as f:
        X_train = pickle.load(f)
    with open( config.get('data_path', 'train_y'), 'rb') as f:
        X_test=pickle.load( f)
    with open(config.get('data_path', 'test_x'), 'rb') as f:
        y_train = pickle.load(f)
    with open(config.get('data_path', 'test_y'), 'rb') as f:
        ytest = pickle.load(f)

    inp = model.input
    outs = {l.name: l.output for l in model.layers}
    predict_fns = {name: K.function([inp], [out]) for name, out in outs.items()}

    # Infering feature ranges.
    features_range = _infer_feature_range(X_test)
    print(len(features_range))

    # Selecting random instances from training set.
    # rnd = np.random.randint(len(X_test) - 101, size=100)

    # lins_layers = {}
    # for name, l in predict_fns.items():
    #     if name != 'input':
    #         def predict_fn(x):
    #             layer = l([x])
    #             return layer[0]
    #         if name == 'softmax':
    #             lins_layers[name] = linearity_measure(predict_fn, X_test[rnd], feature_range=features_range,
    #                                                 agg='global', model_type='regressor', nb_samples=20)
    #         else:
    #             lins_layers[name] = linearity_measure(predict_fn, X_test[rnd], feature_range=features_range, 
    #                                                 agg='global', model_type='regressor', nb_samples=20)
    # lins_layers_mean = {k: v.mean() for k, v in lins_layers.items()}
    # S = pd.Series(data=lins_layers_mean)

    # colors = ['gray' for l in S[:-1]]
    # colors.append('r')
    # ax = S.plot(kind='bar', linewidth=3, figsize=(15,10), color=colors, width=0.7, fontsize=18)
    # ax.set_ylabel('L measure', fontsize=20)
    # ax.set_xlabel('Layer', fontsize=20)
    # print('Linearity measure calculated taking as output each layer of a convolutional neural network.')
    # plt.show()

    class_groups = [X_test]
    def predict_fn(x):
        return model.predict(x)
    lins_classes = []
    t_0 = time()
    for j in range(len(class_groups)):
        print(f'Calculating linearity for instances belonging to class {j}')
        class_group = class_groups[j]
        class_group = np.random.permutation(class_group)
        t_i = time()
        lin = linearity_measure(predict_fn, class_group, feature_range=features_range,
                                agg='global', model_type='regressor')
        t_i_1 = time() - t_i
        print(f'Run time for class {j}: {t_i_1}')
        lins_classes.append(lin)
    t_fin = time() - t_0
    print(f'Total run time: {t_fin}')
    df = pd.DataFrame(data=lins_classes).T

    # ax = df.mean().plot(kind='bar', linewidth=3, figsize=(15,10), color='gray', width=0.7, fontsize=10)
    # ax.set_ylabel('L measure', fontsize=20)
    # ax.set_xlabel('Class', fontsize=20)
    # print("Linearity measure distribution means for each class in the fashion MNIST data set.")
    # plt.show()

    ax2 = df.plot(kind='hist', subplots=True, bins=20, figsize=(10,10), sharey=True)
    for a in ax2:
        a.set_xlabel('L measure', fontsize=20)
        # a.set_ylabel('', rotation=True, fontsize=10)
        a.set_ylabel('F', fontsize=10)
    print('Linearity measure distributions for each class in the fashion MNIST data set.')

    if not os.path.exists(config.get('vis','vis_path_folder11')):
        os.makedirs( config.get('vis','vis_path_folder11'))
    plt.savefig(config.get('vis','vis_path_folder11') + '/linearity_measure.png')

    # plt.show()

linearity_measure_explainer_function()