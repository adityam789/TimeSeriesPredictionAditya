import configparser as cp
import pickle
from keras.models import load_model
import shap
import os
import matplotlib.pyplot as plt

def shap_global_explainer_function():

    model = load_model('Results\model_paths\stock_lstm_model.h5')

    config = cp.RawConfigParser()
    config.read('config/config.properties')
    with open(config.get('data_path', 'train_x'), 'rb') as f:
        X_train = pickle.load(f)
    with open( config.get('data_path', 'train_y'), 'rb') as f:
        X_test=pickle.load( f)

    # rather than use the whole training set to estimate expected values, we summarize with
    # a set of weighted kmeans, each weighted by the number of points they represent.
    X_train_summary = shap.kmeans(X_train.reshape(X_train.shape[0], X_train.shape[1]), 10)

    def f(x):
        x_2 = x.reshape(x.shape[0], x.shape[1], 1)
        return model.predict(x_2)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    explainer = shap.KernelExplainer(f, X_train_summary, link="logit")
    shap_values = explainer.shap_values(X_test)

    with open(config.get('data_path', 'shap_values_global'), 'wb') as f:
    	pickle.dump(shap_values, f)	

    with open(config.get('data_path', 'shap_values_global'), 'rb') as f:
        shap_values = pickle.load(f)

    print(len(shap_values))
    shap.summary_plot(shap_values[-1], X_test, show = False)

    if not os.path.exists(config.get('vis','vis_path_folder8')):
        os.makedirs( config.get('vis','vis_path_folder8'))
    plt.savefig(config.get('vis','vis_path_folder8') + '/shap_global_explain_summary_plot.png')

    print(shap_values[-1][0].reshape(1,100).shape)
    t = shap.force_plot(explainer.expected_value[0], shap_values[-1][0].reshape(1,100), X_test[0], matplotlib = True, show = False)
    if not os.path.exists(config.get('vis','vis_path_folder8')):
        os.makedirs( config.get('vis','vis_path_folder8'))
    plt.savefig(config.get('vis','vis_path_folder8') + '/shap_global_explain_force_plot.png')


    # Things that don't work....
    # print('here')
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, matplotlib = True, show = False)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
    # plt.show()
    # print(explainer.expected_value)
    # print(len(shap_values[0]))
    # t = shap.force_plot(explainer.expected_value[0], shap_values[0],  link="logit", show=True)
    # print(t.keys())
    # print(t)

# shap_global_explainer_function()