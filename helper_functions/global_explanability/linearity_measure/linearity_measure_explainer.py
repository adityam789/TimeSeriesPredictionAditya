from time import time
from alibi.confidence.model_linearity import LinearityMeasure, linearity_measure, _infer_feature_range
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
import pandas as pd
from tensorflow.keras import backend as K

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

lins_dict = {}

print(X_train.shape)
print(y_train.shape)

# # Creating a grid
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# # Flattening points in the grid
# X = np.empty((len(xx.flatten()), 2))
# for i in range(xx.shape[0]):
#     for j in range(xx.shape[1]):
#         k = i * xx.shape[1] + j
#         X[k] = np.array([xx[i, j], yy[i, j]])

# feature_names = [str(i) for i in range(100)]
# target_names = ['a']

# inp = model.input
# outs = {l.name: l.output for l in model.layers}
# predict_fns = {name: K.function([inp], [out]) for name, out in outs.items()}

# # Infering feature ranges.
# features_range = _infer_feature_range(X_test)

# # Selecting random instances from training set.
# rnd = np.random.randint(len(X_test) - 101, size=100)

# lins_layers = {}
# for name, l in predict_fns.items():
#     if name != 'input':
#         def predict_fn(x):
#             layer = l([x])
#             return layer[0]
#         if name == 'softmax':
#             lins_layers[name] = linearity_measure(predict_fn, X_test[rnd], feature_range=features_range,
#                                                   agg='global', model_type='classifier', nb_samples=20)
#         else:
#             lins_layers[name] = linearity_measure(predict_fn, X_test[rnd], feature_range=features_range,
#                                                   agg='global', model_type='regressor', nb_samples=20)
# lins_layers_mean = {k: v.mean() for k, v in lins_layers.items()}
# S = pd.Series(data=lins_layers_mean)

# colors = ['gray' for l in S[:-1]]
# colors.append('r')
# ax = S.plot(kind='bar', linewidth=3, figsize=(15,10), color=colors, width=0.7, fontsize=18)
# ax.set_ylabel('L measure', fontsize=20)
# ax.set_xlabel('Layer', fontsize=20)
# print('Linearity measure calculated taking as output each layer of a convolutional neural network.')

# class_groups = []
# for i in range(10):
#     y = ytest.argmax(axis=0)
#     idxs_i = np.where(y == i)[0]
#     class_groups.append(X_test[idxs_i])

# def predict_fn(x):
#     return model.predict(x)
# lins_classes = []
# t_0 = time()
# for j in range(len(class_groups)):
#     print(f'Calculating linearity for instances belonging to class {j}')
#     class_group = class_groups[j]
#     class_group = np.random.permutation(class_group)[:2000]
#     t_i = time()
#     lin = linearity_measure(predict_fn, class_group, feature_range=features_range,
#                             agg='global', model_type='classifier', nb_samples=20)
#     t_i_1 = time() - t_i
#     print(f'Run time for class {j}: {t_i_1}')
#     lins_classes.append(lin)
# t_fin = time() - t_0
# print(f'Total run time: {t_fin}')

# df = pd.DataFrame(data=lins_classes).T
# ax = df.mean().plot(kind='bar', linewidth=3, figsize=(15,10), color='gray', width=0.7, fontsize=10)
# ax.set_ylabel('L measure', fontsize=20)
# ax.set_xlabel('Class', fontsize=20)
# print("Linearity measure distribution means for each class in the fashion MNIST data set.")

# ax2 = df.plot(kind='hist', subplots=True, bins=20, figsize=(10,10), sharey=True)
# for a in ax2:
#     a.set_xlabel('L measure', fontsize=20)
#     a.set_ylabel('', rotation=True, fontsize=10)
# #ax2.set_ylabel('F', fontsize=10)
# print('Linearity measure distributions for each class in the fashion MNIST data set.')


# predict_fn = lambda x: model.predict(x)

# print(X.shape)

# lm = LinearityMeasure(agg='pairwise')
# lm.fit(X_train)
# L = lm.score(predict_fn, X)
# L = L.reshape(xx.shape)
# lins_dict['NN'] = L.mean()

# feature_range = _infer_feature_range(X_train)
# L = linearity_measure(predict_fn,
#                       X,
#                       feature_range=feature_range,
#                       method='grid',
#                       X_train=None,
#                       epsilon=0.04,
#                       nb_samples=10,
#                       res=100,
#                       alphas=None,
#                       agg='global',
#                       model_type='regressor')

# Visualising decision boundaries and linearity values
# f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(16, 8))
# idx = (0,0)
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# axarr[0].contourf(xx, yy, Z, alpha=0.4)
# axarr[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', alpha=1)
# axarr[0].set_title('Decision boundaries', fontsize=20)
# axarr[0].set_xlabel('sepal length (cm)', fontsize=18)
# axarr[0].set_ylabel('sepal width (cm)', fontsize=18)

# LPL = axarr[1].contourf(xx, yy, L, alpha=0.8, cmap='Greys')
# axarr[1].set_title('Model linearity', fontsize=20)
# axarr[1].set_xlabel('sepal length (cm)', fontsize=18)
# axarr[1].set_ylabel('sepal width (cm)', fontsize=18)

# cbar = f.colorbar(LPL)
# #cbar.ax.set_ylabel('Linearity')
# plt.show()
# print('Decision boundaries (left panel) and linearity measure (right panel) for a feed forward neural network classifier (NN) classifier in feature space. The x and y axis in the plots represent the sepal length and the sepal width, respectively.  Different colours correspond to different predicted classes. The markers represents the data points in the training set.')
# print('Maximum value model linearity: {}'. format(np.round(L.max(), 5)))
# print('Minimum value model linearity: {}'.format(np.round(L.min(),5)))