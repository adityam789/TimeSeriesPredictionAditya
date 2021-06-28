# import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import math
import pandas as pd
import configparser as cp
import os
import pickle
import matplotlib.pyplot as plt
# from .Detectors.adwin import *
# from .Detectors.ddm import DDM
from skmultiflow.drift_detection.adwin import ADWIN
adwin = ADWIN()
plt.style.use('fivethirtyeight')

config = cp.RawConfigParser()
config.read('config/config.properties')

 # convert an array of values into a dataset matrix


def create_dataset(dataset, time_step=1):

    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def change_detector_adwin(df, training_size):
    print(training_size)
    change_detected = []
    train_data_change_detected = [0]
    test_data_change_detected = [training_size]
    for i in range(df.size):
        if i < training_size:
            adwin.add_element(df.iat[i, 0])
            # if adwin.detected_warning_zone():
            #     print('Warning detected in data: ' + str(df.iat[i,0]) + ' - at index: ' + str(i))
            if adwin.detected_change():
                train_data_change_detected.append(i)
                change_detected.append(i)
                print('Change detected in data: ' +
                      str(df.iat[i, 0]) + ' - at index: ' + str(i))
        else:
            adwin.add_element(df.iat[i, 0])
            # if adwin.detected_warning_zone():
            #     print('Warning detected in data: ' + str(df.iat[i,0]) + ' - at index: ' + str(i))
            if adwin.detected_change():
                test_data_change_detected.append(i)
                change_detected.append(i)
                print('Change detected in data: ' +
                      str(df.iat[i, 0]) + ' - at index: ' + str(i))
    if train_data_change_detected[-1] != training_size:
        train_data_change_detected += [training_size]
    test_data_change_detected += [df.size]
    print(train_data_change_detected)
    return train_data_change_detected, test_data_change_detected, change_detected


def preprocessing():

	df = pd.read_csv(config.get('dataset_path', 'csv_path'))
	df1 = df[[config.get('target_column', 'target_column')]]
	df2 = df[[config.get('target_column', 'target_column')]]

	# garaph
	target_column = config.get('target_column', 'target_column')
	df_10 = pd.DataFrame()
	df_10[target_column] = df[target_column].rolling(window=10).mean()
	df_20 = pd.DataFrame()
	df_20[target_column] = df[target_column].rolling(window=20).mean()
	df_30 = pd.DataFrame()
	df_30[target_column] = df[target_column].rolling(window=30).mean()
	df_40 = pd.DataFrame()
	df_40[target_column] = df[target_column].rolling(window=40).mean()

	# Visualize the data
	plt.figure(figsize=(8, 4))
	plt.plot(df[target_column].tail(200), label='df')
	plt.plot(df_10[target_column].tail(200), label='df_10')
	plt.plot(df_20[target_column].tail(200), label='df_20')
	# plt.plot(df_30[target_column].tail(200), label='df_30')
	# plt.plot(df_40[target_column].tail(200), label='df_40')
	plt.title('Apple Close Price History')
	plt.xlabel('Mar. 23, 2008 - Nov. 10, 2017')
	plt.ylabel('Close Price USD($)')
	plt.legend(loc='upper left')

	if not os.path.exists(config.get('vis', 'vis_path_folder')):
		os.makedirs(config.get('vis', 'vis_path_folder'))
	plt.savefig(config.get('vis', 'vis_path_folder') + '/mean_plot.png')

	scaler = MinMaxScaler(feature_range=(0, 1))
	df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
	training_size = int(
	    len(df1)*config.getfloat('features', 'train_data_percent'))
	test_size = len(df1)-training_size
	train_data, test_data = df1[0:training_size,
	    :], df1[training_size:len(df1), :1]

	# train_data,test_data =preprocessing()
	time_step = config.getint('features', 'time_step')
	X_train, y_train = create_dataset(train_data, time_step)
	X_test, ytest = create_dataset(test_data, time_step)
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
	
	train_data_change_detected, test_data_change_detected, drifts_detected = change_detector_adwin(
	    df2, X_train.shape[0])

	if not os.path.exists(config.get('data_path', 'data_path_folder')):
		os.makedirs(config.get('data_path', 'data_path_folder'))

	with open(config.get('data_path', 'train_x'), 'wb') as f:
		pickle.dump(X_train, f)
	with open(config.get('data_path', 'train_y'), 'wb') as f:
		pickle.dump(X_test, f)
	with open(config.get('data_path', 'test_x'), 'wb') as f:
		pickle.dump(y_train, f)
	with open(config.get('data_path', 'test_y'), 'wb') as f:
		pickle.dump(ytest, f)
	with open(config.get('data_path', 'scaler_dump'), 'wb') as f:
		pickle.dump(scaler, f)
	with open(config.get('data_path', 'df1'), 'wb') as f:
		pickle.dump(df1, f)
	with open(config.get('data_path', 'test_data_change_detected_ADWIN'), 'wb') as f:
		pickle.dump(test_data_change_detected, f)
	with open(config.get('data_path', 'train_data_change_detected_ADWIN'), 'wb') as f:
		pickle.dump(train_data_change_detected, f)	
	with open(config.get('data_path', 'change_detected_ADWIN'), 'wb') as f:
		pickle.dump(drifts_detected, f)	
	# dump(scaler, open('scaler.pkl', 'wb'))

	print(df1.shape)
	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(ytest.shape)
	print(training_size)

# preprocessing()



 

