# from __future__ import print_function
import numpy as np
# np.random.seed(1)
# import sklearn.ensemble
# from anchor import utils
# from anchor import anchor_tabular
from alibi.explainers import AnchorTabular
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import numpy as np  
import os
from keras.models import load_model
import pandas as pd

model = load_model('Results\model_paths\stock_lstm_model.h5')

config = cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'train_x'), 'rb') as f:
    X_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
    X_test=pickle.load( f)

# feature_names = [str(i) for i in range(100)]
# target_names = ['a']

# explainer = anchor_tabular.AnchorTabularExplainer(
#     target_names,
#     feature_names,
#     X_train
#     # dataset.categorical_names
# )

# idx = 0
# np.random.seed(1)
# print('Prediction: ', explainer.class_names[model.predict(X_test[idx])])
# exp = explainer.explain_instance(X_test[idx], model.predict, threshold=0.95)

from alibi.utils.data import gen_category_map

df = pd.DataFrame(data=X_train.reshape(X_train.shape[0], X_train.shape[1]), columns = [str(i) for i in range(100)])

category_map = gen_category_map(df)

feature_names = [str(i) for i in range(100)]
target_names = ['a']

np.random.seed(0)
data_perm = np.random.permutation(np.c_[X_train, target_names])
data = data_perm[:,:-1]
target = data_perm[:,-1]
idx = 100
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]

predict_fn = lambda x: model.predict(x)
explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map)
explainer.fit(X_train, disc_perc=[25, 50, 75])
idx = 0
class_names = target_names
print('Prediction: ', class_names[explainer.predictor(X_test[idx].reshape(1, -1))[0]])
explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)