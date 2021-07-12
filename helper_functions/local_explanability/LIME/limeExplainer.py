from lime import lime_tabular
import configparser as cp
import pickle
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import numpy as np

def instance_plotter(index, explainer, X_train, model, path):

    config=cp.RawConfigParser()
    config.read('config/config.properties')
    print(index)

    exp = explainer.explain_instance(
        data_row = X_train[index].reshape(1,100,1),
        classifier_fn = model.predict,
        num_features = 100)

    weights = exp.as_list()
    minV, maxV = [], []
    t = [i[1] for i in weights]
    n = np.array(t)
    n_copy = n.copy()

    for i in range(10):
        j = np.argmin(n)
        minV.append((j,weights[j][1]))
        n[j] = 10000000000

    for m in range(10):
        k = np.argmax(n_copy)
        maxV.append((k,weights[k][1]))
        n_copy[k] = -10000000000

    minNames = [i[0] for i in minV]
    minValues = [i[1] for i in minV]
    maxNames = [i[0] for i in maxV]
    maxValues = [i[1] for i in maxV]
    minColors = ['green' if x > 0 else 'red' for x in minValues]
    maxColors = ['green' if x > 0 else 'red' for x in maxValues]
    figure, axes = plt.subplots(nrows=1, ncols=2)
    figure.figsize= (16,8)
    pos = np.arange(10)

    axes[0].barh(pos, minValues, align='center', color=minColors)
    axes[0].set_yticks(pos, minor=False)
    axes[0].set_yticklabels(minNames, fontdict=None, minor=False)
    axes[0].set_title('Lime Explaination (10 Min)')
    axes[0].set_xlabel('Feature Contribution')
    axes[0].set_ylabel('Days or Features of input')

    axes[1].barh(pos, maxValues, align='center', color=maxColors)
    axes[1].set_yticks(pos, minor=False)
    axes[1].set_yticklabels(maxNames, fontdict=None, minor=False)
    axes[1].set_title('Lime Explaination (10 Max)')
    axes[1].set_xlabel('Feature Contribution')
    axes[1].set_ylabel('Days or Features of input')

    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder4')):
        os.makedirs( config.get('vis','vis_path_folder4'))
    plt.savefig(config.get('vis','vis_path_folder4') + path)

    # plt.show()

    # fig = exp.as_pyplot_figure()    
    # fig.figsize = (12,6)
    # plt.title('Lime Explaination')
    # plt.xlabel('Feature Contribution')
    # plt.ylabel('Days or Features of input (With discretizer range)')

    # #saving graphic
    # if not os.path.exists(config.get('vis','vis_path_folder3')):
    #     os.makedirs( config.get('vis','vis_path_folder3'))
    # plt.savefig(config.get('vis','vis_path_folder3') + '/lime_explain.png')

    # plt.show()

def lime_explainer_function():

    model = load_model('Results\model_paths\stock_lstm_model.h5')

    config=cp.RawConfigParser()
    config.read('config/config.properties')

    with open(config.get('data_path', 'train_x'), 'rb') as f:
        X_train = pickle.load(f)
    with open(config.get('data_path', 'test_x'), 'rb') as f:
        y_train =pickle.load(f)
    print(os.path.getsize(config.get('data_path', 'min10errors')))
    with open(config.get('data_path', 'min10errors'), 'rb') as f:
        mins = pickle.load(f)
    with open(config.get('data_path', 'max10errors'), 'rb') as f:
        maxs = pickle.load(f)

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

    for i, j in enumerate(mins):
        path = "/lime_explain_least_" + str(i) + ".png"
        instance_plotter(j[1], explainer, X_train, model, path)
        print("Done "+str(i)+" rounds of deep explain")

    for i, j in enumerate(maxs):
        path = "/lime_explain_highest_" + str(i) + ".png"
        instance_plotter(j[1], explainer, X_train, model, path)
        print("Done "+str(i)+" rounds of deep explain")

    print("Done explaining with lime")
