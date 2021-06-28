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
from .Detectors.drift_detector import *
from .Graph.graph_execute import *
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

def preprocessing():

	df = pd.read_csv(config.get('dataset_path', 'csv_path'))
	df1 = df[[config.get('target_column', 'target_column')]]
	df2 = df[[config.get('target_column', 'target_column')]]

	# graph the moving average
	graphMovingAverage(df)

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
	
	train_data_change_detected, test_data_change_detected, drifts_detected = detector(
	    df2, X_train.shape[0], "ADWIN")

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



 

