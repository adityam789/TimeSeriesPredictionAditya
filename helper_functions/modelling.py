# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
import math
import pandas as pd
import configparser as cp
import pickle
import os 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#plt.style.use('fivethirtyeight')
config=cp.RawConfigParser()
config.read('config/config.properties')

def model_making():

	with open( config.get('data_path', 'train_x'), 'rb') as f:
		X_train=pickle.load(f)
	with open( config.get('data_path', 'train_y'), 'rb') as f:
		X_test=pickle.load( f)
	with open(config.get('data_path', 'test_x'), 'rb') as f:
 		y_train =pickle.load(f)
	with open(config.get('data_path', 'test_y'), 'rb') as f:
 		ytest=pickle.load(f)
	with open(config.get('data_path', 'scaler_dump'), 'rb') as f:
 		scaler=pickle.load(f)
	with open(config.get('data_path', 'df1'), 'rb') as f:
 		df1=pickle.load(f)
	with open(config.get('data_path', 'test_data_change_detected_ADWIN'), 'rb') as f:
 		test_data_change_detected_ADWIN=pickle.load(f)
	with open(config.get('data_path', 'train_data_change_detected_ADWIN'), 'rb') as f:
 		train_data_change_detected_ADWIN=pickle.load(f)


	# Model 1 without Concept Drift (With Test Data) 

	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))   
	# model.add(LSTM(50, return_sequences=True, input_shape=(100,1))) 
	model.add(LSTM(50, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	print( model.summary())

	model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=config.getint('model_params','epochs'),batch_size=config.getint('model_params','batch_size'),verbose=1)
	train_predict=model.predict(X_train)
	test_predict=model.predict(X_test)

	# print(math.sqrt(mean_squared_error(y_train,train_predict)),math.sqrt(mean_squared_error(ytest,test_predict)))
	# print(math.sqrt(mean_squared_error(ytest,test_predict)))   
	# print(mean_absolute_error(ytest,test_predict))
	# print( r2_score(ytest,test_predict))
	# train_predict=scaler.inverse_transform(train_predict)
	# test_predict=scaler.inverse_transform(test_predict)

	# Model 2 with Concept Drift (Without Test Data) 

	model2 = Sequential()
	model2.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))   
	# model.add(LSTM(50, return_sequences=True, input_shape=(100,1))) 
	model2.add(LSTM(50, return_sequences=False))
	model2.add(Dense(25))
	model2.add(Dense(1))
	model2.compile(optimizer='adam', loss='mean_squared_error')
	print( model2.summary())

	predict1 = np.array([[1]])
	predict2 = np.array([[1]])
	for k in range(len(train_data_change_detected_ADWIN)-1):
		X_batch = X_train[train_data_change_detected_ADWIN[k]:train_data_change_detected_ADWIN[k+1]]
		y_batch = y_train[train_data_change_detected_ADWIN[k]:train_data_change_detected_ADWIN[k+1]]
		y_pred = model2.predict_on_batch(X_batch)
		predict1 = np.concatenate((predict1, y_pred), axis=0)
		up_y = y_batch
		a_score = math.sqrt(mean_squared_error(y_batch.flatten(), y_pred.flatten()))
		w = model2.layers[0].get_weights() #Only get weights for LSTM layer
		for l in range(len(w)):
			w[l] = w[l] - (w[l]*0.001*a_score) #0.001=learning rate
		model2.layers[0].set_weights(w)
		model2.fit(X_batch, up_y, epochs=10, verbose=1)
		pred2 = model2.predict_on_batch(X_batch)
		predict2 = np.concatenate((predict2, pred2), axis=0)
		model2.reset_states()
	
	predict1 = np.delete(predict1,0,0)
	predict2 = np.delete(predict2,0,0)

	print(math.sqrt(mean_squared_error(y_train,predict2.flatten())))
	# print(math.sqrt(mean_squared_error(y_train,train_predict)),math.sqrt(mean_squared_error(ytest,test_predict)))
	# print(math.sqrt(mean_squared_error(ytest,test_predict)))   
	# print(mean_absolute_error(ytest,test_predict))
	# print( r2_score(ytest,test_predict))
	# train_predict=scaler.inverse_transform(train_predict)
	# test_predict=scaler.inverse_transform(test_predict)


	models_path_folder = config.get('models_path','models_path_folder')

	if not os.path.exists( models_path_folder):
		os.makedirs(models_path_folder)

	model_name = config.get('models_path','model_name')
	model.save(models_path_folder+ model_name)

	#shift train predictions for plotting
	## Plotting 
    #vis of graphs
	look_back=X_train.shape[1]
	trainPredictPlot = np.empty_like(df1)		
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(df1)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
	# Visualize the data
	plt.figure(figsize=(8,4))
	plt.title('Model')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(df1),label='True')
	plt.plot(trainPredictPlot,label='Train')
	plt.plot(testPredictPlot,label='Test')		
	if not os.path.exists(config.get('vis','vis_path_folder')):
		os.makedirs( config.get('vis','vis_path_folder'))
	plt.savefig(config.get('vis','vis_path_folder') + '/model_performance.png')
  
	#..........................


	# demonstrate prediction for next 10 days
	n_steps=config.getint('features','time_step')+1
	prediction_next_days=config.getint('prediction','prediction_next_days')
	x_input=df1[-(n_steps):].reshape(1,-1)
	temp_input=list(x_input)
	temp_input=temp_input[0].tolist()
	lst_output=[]
	# n_steps=100
	i=0
	while(i<prediction_next_days):
	    
	    if(len(temp_input)>n_steps):
	        #print(temp_input)
	        x_input=np.array(temp_input[1:])
	        #print("{} day input {}".format(i,x_input))
	        x_input=x_input.reshape(1,-1)
	        x_input = x_input.reshape((1, n_steps, 1))
	        #print(x_input)
	        yhat = model.predict(x_input, verbose=0)
	        #print("{} day output {}".format(i,yhat))
	        temp_input.extend(yhat[0].tolist())
	        temp_input=temp_input[1:]
	        #print(temp_input)
	        lst_output.extend(yhat.tolist())
	        i=i+1
	    else:
	        x_input = x_input.reshape((1, n_steps,1))
	        yhat = model.predict(x_input, verbose=0)
	        #print(yhat[0])
	        temp_input.extend(yhat[0].tolist())
	        #print(len(temp_input))
	        lst_output.extend(yhat.tolist())
	        i=i+1
	lst_output=np.array(lst_output)
	day_new=np.arange(1,101)
	day_pred=np.arange(101,101+prediction_next_days)
	plt.figure(figsize=(8,4))
	plt.title('Model')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	plt.legend(['True', 'pred'],loc='upper left')
	plt.plot(day_new,scaler.inverse_transform(df1[-100:]))
	plt.plot(day_pred,scaler.inverse_transform(lst_output))

	if not os.path.exists(config.get('vis','vis_path_folder')):
		os.makedirs( config.get('vis','vis_path_folder'))
	plt.savefig(config.get('vis','vis_path_folder') + '/model_predection.png')

				



#model_making()