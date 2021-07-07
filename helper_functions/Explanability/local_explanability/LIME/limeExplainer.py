from lime import lime_tabular
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import os

from keras.models import load_model
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

fig = exp.as_pyplot_figure()    

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder3')):
    os.makedirs( config.get('vis','vis_path_folder3'))
plt.savefig(config.get('vis','vis_path_folder3') + '/lime_explain.png')

plt.show()