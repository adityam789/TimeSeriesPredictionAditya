from lime import lime_tabular
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import os

from keras.models import load_model
import numpy as np
model = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'train_x'), 'rb') as f:
	X_train = pickle.load(f)
with open(config.get('data_path', 'test_x'), 'rb') as f:
 	y_train =pickle.load(f)

print(X_train.shape)
print(y_train.shape)

explainer = lime_tabular.RecurrentTabularExplainer(
    X_train,
    mode="regression",
    training_labels = y_train,
    feature_names = ['1']) # ,'2','3','4','5','6','7','8','9','10','11','12'],
    # discretize_continuous = False
    # # class_names = ['a','b','c','d','e','f','g','h','i'],
    # discretizer = 'decile')

exp = explainer.explain_instance(
    data_row = X_train[100].reshape(1,100,1),
    classifier_fn = model.predict,
    num_features = 100)

weights = exp.as_list()
minV, maxV = [], []
t = [i[1] for i in weights]
n = np.array(t)
for i in range(10):
    j = np.argmin(n)
    minV.append(weights[j])
    n = np.delete(n, j)

for m in range(10):
    k = np.argmax(n)
    maxV.append(weights[k])
    n = np.delete(n, k)

minNames = [i[0] for i in minV]
minValues = [i[1] for i in minV]
maxNames = [i[0] for i in maxV]
maxValues = [i[1] for i in maxV]
minColors = ['green' if x > 0 else 'red' for x in minValues]
maxColors = ['green' if x > 0 else 'red' for x in maxValues]
print(maxNames)
print(maxValues)
figure, axes = plt.subplots(nrows=1, ncols=2)
figure.figsize= (12,6)
pos = np.arange(10)

fontdictlol = {'fontsize': 20,
 'fontweight': 'bold',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'loc'}

axes[0].barh(pos, minValues, align='center', color=minColors)
axes[0].set_yticks(pos, minor=False)
axes[0].set_yticklabels(minNames, fontdict=None, minor=False)
axes[0].set_title('Lime Explaination (10 Min)')
axes[0].set_xlabel('Feature Contribution')
axes[0].set_ylabel('Days or Features of input (With discretizer range)')

axes[1].barh(pos, maxValues, align='center', color=maxColors)
axes[1].set_yticks(pos, minor=False)
axes[1].set_yticklabels(maxNames, fontdict=fontdictlol, minor=False)
axes[1].set_title('Lime Explaination (10 Max)')
axes[1].set_xlabel('Feature Contribution')
axes[1].set_ylabel('Days or Features of input (With discretizer range)')

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder3')):
    os.makedirs( config.get('vis','vis_path_folder3'))
plt.savefig(config.get('vis','vis_path_folder3') + '/lime_explain_minmax.png')

plt.show()

fig = exp.as_pyplot_figure()    
fig.figsize = (12,6)
plt.title('Lime Explaination')
plt.xlabel('Feature Contribution')
plt.ylabel('Days or Features of input (With discretizer range)')

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder3')):
    os.makedirs( config.get('vis','vis_path_folder3'))
plt.savefig(config.get('vis','vis_path_folder3') + '/lime_explain.png')

plt.show()