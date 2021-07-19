import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
import pandas as pd
import configparser as cp
import pickle
import os
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from .Graph.graph_execute import *

config = cp.RawConfigParser()
config.read('config/config.properties')


def model_making():

    with open(config.get('data_path', 'train_x'), 'rb') as f:
        X_train = pickle.load(f)
    with open(config.get('data_path', 'train_y'), 'rb') as f:
        X_test = pickle.load(f)
    with open(config.get('data_path', 'test_x'), 'rb') as f:
        y_train = pickle.load(f)
    with open(config.get('data_path', 'test_y'), 'rb') as f:
        ytest = pickle.load(f)
    with open(config.get('data_path', 'scaler_dump'), 'rb') as f:
        scaler = pickle.load(f)
    with open(config.get('data_path', 'df1'), 'rb') as f:
        df1 = pickle.load(f)
    with open(config.get('data_path', 'test_data_change_detected_ADWIN'), 'rb') as f:
        test_data_change_detected_ADWIN = pickle.load(f)
    with open(config.get('data_path', 'train_data_change_detected_ADWIN'), 'rb') as f:
        train_data_change_detected_ADWIN = pickle.load(f)

    # Model 1 without Concept Drift (With Test Data)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['accuracy', 'binary_crossentropy'])
    print(model.summary())

    start1 = time.time()
    hist_model = model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=config.getint(
        'model_params', 'epochs'), batch_size=config.getint('model_params', 'batch_size'), verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    end1 = time.time()

    # Scores
    RMSE2 = math.sqrt(mean_squared_error(y_train, train_predict))
    
    # print(math.sqrt(mean_squared_error(y_train,train_predict)),math.sqrt(mean_squared_error(ytest,test_predict)))
    # print(math.sqrt(mean_squared_error(ytest,test_predict)))
    # print(mean_absolute_error(ytest,test_predict))
    # print( r2_score(ytest,test_predict))

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Model 2 with Concept Drift (Without Test Data)

    model2 = Sequential()
    model2.add(LSTM(50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
    model2.add(LSTM(50, return_sequences=False))
    model2.add(Dense(25))
    model2.add(Dense(1))
    model2.compile(optimizer='adam', loss='mean_squared_error',
                   metrics=['accuracy', 'binary_crossentropy'])
    print(model2.summary())

    predict1 = np.array([[1]])
    predict2 = np.array([[1]])
    start2 = time.time()
    for k in range(len(train_data_change_detected_ADWIN)-1):
        X_batch = X_train[train_data_change_detected_ADWIN[k]:train_data_change_detected_ADWIN[k+1]]
        y_batch = y_train[train_data_change_detected_ADWIN[k]:train_data_change_detected_ADWIN[k+1]]
        y_pred = model2.predict_on_batch(X_batch)
        predict1 = np.concatenate((predict1, y_pred), axis=0)
        up_y = y_batch
        a_score = math.sqrt(mean_squared_error(
            y_batch.flatten(), y_pred.flatten()))
        w = model2.layers[0].get_weights()  # Only get weights for LSTM layer
        for l in range(len(w)):
            w[l] = w[l] - (w[l]*0.001*a_score)  # 0.001=learning rate
        model2.layers[0].set_weights(w)
        hist_model2 = model2.fit(X_batch, up_y, epochs=config.getint(
            'model_params', 'epochs'), verbose=1)
        pred2 = model2.predict_on_batch(X_batch)
        predict2 = np.concatenate((predict2, pred2), axis=0)
        model2.reset_states()
    end2 = time.time()
    predict1 = np.delete(predict1, 0, 0)
    predict2 = np.delete(predict2, 0, 0)

    # Scores
    RMSE = math.sqrt(mean_squared_error(y_train, predict2.flatten()))
    print(RMSE)
    print(RMSE2)
    print(end1 - start1)
    print(end2 - start2)

    # print("Line 109 ", model.evaluate(X_test, ytest))
    # print("Line 110 ", model2.evaluate(X_test, ytest))
    # print("Line 111 ", hist_model.history['accuracy'])
    # print("Line 112 ", hist_model2.history)

    models_path_folder = config.get('models_path', 'models_path_folder')

    if not os.path.exists(models_path_folder):
        os.makedirs(models_path_folder)

    model_name = config.get('models_path', 'model_name')
    model.save(models_path_folder + model_name)

    # Plot Shobit data:

    plotGraphShobit(X_train, df1, train_predict, test_predict, scaler)

    # ..........................

    def keyOfTuple(t):
        return t[0]

    # Plot Aditya data:
    error_stream = []
    error_stream2 = []
    error_stream_with_indexes = []
    error_stream2_with_indexes = []
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    predict2 = scaler.inverse_transform(predict2)
    for i in range(y_train.size):
        error_stream2.append(abs(y_train[i] - train_predict[i][0]))
        error_stream.append(abs(y_train[i] - predict2[i][0]))
        error_stream_with_indexes.append((abs(y_train[i] - predict2[i][0]), i))
        error_stream2_with_indexes.append(
            (abs(y_train[i] - train_predict[i][0]), i))

    sorted_Errors = sorted(error_stream2_with_indexes, key=keyOfTuple)
    maxVs = sorted_Errors[-5:]
    minVs = sorted_Errors[:5]
    print(minVs, maxVs)
    with open(config.get('data_path', 'min10errors'), 'wb') as f:
        pickle.dump(minVs, f)
    with open(config.get('data_path', 'max10errors'), 'wb') as f:
        pickle.dump(maxVs, f)

    Plot_graph_series(y_train, predict2, train_data_change_detected_ADWIN,
                      10, RMSError=RMSE, name=config.get('graph_labels', 'data_title'))
    plotGraphError(y_train, train_data_change_detected_ADWIN,
                   error_stream, 10, config.get('graph_labels', 'data_title'))

    # ..........................

    # Plot Comparison data:
    plot_difference(train_predict, predict2, y_train)

    plot_difference_comparison(error_stream2, error_stream)
    # ..........................

    # demonstrate prediction for next 10 days
    n_steps = config.getint('features', 'time_step')+1
    prediction_next_days = config.getint('prediction', 'prediction_next_days')
    x_input = df1[-(n_steps):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    # desc = "This is a graph illustrating the future prediction of model.\nThe plot showcases the data of past " + str(n_steps - 1) + " indexes (days) and predicts the future values for the next 10 indexes (days).\nThe x-axis represents the indexes of the data and the y-axis represents the data name/ type."
    n_steps=100
    i = 0
    while(i < prediction_next_days):

        if(len(temp_input) > n_steps):
            # print(temp_input)
            x_input = np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            # print(temp_input)
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            # print(x_input.shape)
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i+1
    lst_output = np.array(lst_output)
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 101+prediction_next_days)
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.1)
    plt.suptitle(config.get('graph_labels', 'data_title'))
    plt.title('Future Prediction')
    plt.xlabel('Index', fontsize=18)
    plt.ylabel(config.get('graph_labels', 'data_type'), fontsize=18)
    # plt.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    # plt.annotate(desc, (0,0), (-60, -40), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.legend(['True', 'pred'], loc='upper left')
    plt.plot(day_new, scaler.inverse_transform(df1[-100:]))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))

    if not os.path.exists(config.get('vis', 'vis_path_folder2')):
        os.makedirs(config.get('vis', 'vis_path_folder2'))
    plt.savefig(config.get('vis', 'vis_path_folder2') +
                '/model_predection.png')


# model_making()
