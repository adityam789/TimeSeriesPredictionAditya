import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configparser as cp
import pickle
import os 
plt.style.use('seaborn-white')

config=cp.RawConfigParser()
config.read('config/config.properties')

def Plot_graph_series(stream, prediction_vector, detections, n, alarms=None,  delays=None, false_alarms=None, execution_time=None, RMSError=None, hitRatio=None, name=None):

    # style of some graphics functions
    detector_width = 2
    style = 'dashed'
    color_data_real = 'Blue'
    prediction_color = 'Red'
    
    detection_found_color = '#00BFFF'
    retraining_color = 'Gray'
    online_color = 'Yellow'
    alpha_retraining_color = 0.15

    X_axis = [0] * 2
    X_axis[0] = 0
    X_axis[1] = len(stream)
    Y_axis_error = [0] * 2
    Y_axis_error[0] = 0
    Y_axis_error[1] = 0.2
    X_intervals = range(X_axis[0], X_axis[1], 900)

    label_detection_found = 'Drift Found'

    # Creating a figure
    figure = plt.figure()
    if(name != None):
        figure.suptitle(name, fontsize=11, fontweight='bold')

    # Defining Plot 1 with preview and actual data
    graph1 = figure.add_subplot()
    graph1.plot(stream, label='Original Series', color=color_data_real, linewidth=0.5)
    graph1.plot(prediction_vector, label='Forecast', color=prediction_color, linewidth=0.5)

    if(RMSError == None):
        graph1.set_title("Real dataset and forecast")
    else:
        #graph1.set_title("Real dataset and forecast | RMSError: %.3f | hit ratio: %.3f |"  % (RMSError, hitRatio))
        graph1.set_title("Real dataset and forecast | RMSError: %.3f |" % (RMSError))

    # Inserting Labels
    graph1.axvline( linewidth=detector_width, linestyle=style, label=label_detection_found, color=detection_found_color)

    # plotting Detections In Error Graphics
    for i in range(len(detections)):
        counter = detections[i]
        graph1.axvline(counter, linewidth=detector_width,
                       linestyle=style, color=detection_found_color)
        graph1.axvspan(counter, counter+n, facecolor=retraining_color,
                       alpha=alpha_retraining_color, zorder=10)
        if(i > 0):
            graph1.axvspan(detections[i-1]+n, counter, facecolor=online_color, alpha=alpha_retraining_color)

    graph1.axvspan(
        0, detections[0], facecolor=online_color, alpha=alpha_retraining_color)
    graph1.axvspan(detections[len(detections)-1]+n, X_axis[1],
                   facecolor=online_color, alpha=alpha_retraining_color)

    # putting caption and defining the graphic axes
    plt.ylabel('Observations')
    plt.xlabel('Index')
    # graph1.axis([X_axis[0], X_axis[1], 0, 1])
    graph1.legend(loc='upper center', ncol=1, fancybox=True, shadow=True)
    # plt.xticks(X_intervals, rotation=45)

    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder')):
        os.makedirs( config.get('vis','vis_path_folder'))
    plt.savefig(config.get('vis','vis_path_folder') + '/model_performance_CD.png')

    #showing graphic
    plt.show()


def plotGraphError(stream, detections, error_stream_vector, n, name):

    print('hero')

    error_color = 'Blue'
    detector_width = 2
    style = 'dashed'
    label_detection_found = 'Drift Found'

    detection_found_color = '#00BFFF'
    alpha_retraining_color = 0.15
    retraining_color = 'Gray'

    X_axis = [0] * 2
    X_axis[0] = 0
    X_axis[1] = len(stream)
    Y_axis_error = [0] * 2
    Y_axis_error[0] = 0
    Y_axis_error[1] = 0.2
    X_intervals = range(X_axis[0], X_axis[1], 900)

    # Creating a figure
    figure2 = plt.figure()
    if(name != None):
        figure2.suptitle(name, fontsize=11, fontweight='bold')

    # Defining Plot 2 with the forbearing error
    grafico2 = figure2.add_subplot()
    grafico2.plot(error_stream_vector,
                  label='Forecasting Error', color=error_color, linewidth=0.5)
    grafico2.set_title("Forecasting Error")

    # Defining Labels
    grafico2.axvline(linewidth=detector_width, linestyle=style,
                     label=label_detection_found, color=detection_found_color)

    # Plotting detection in error graphics
    for i in range(len(detections)):
        counter = detections[i]
        grafico2.axvline(counter, linewidth=detector_width,
                         linestyle=style, color=detection_found_color)
        grafico2.axvspan(counter, counter+n, facecolor=retraining_color,
                         alpha=alpha_retraining_color, zorder=10)

    # putting caption and defining the graphic axes
    plt.ylabel('RMSError')
    plt.xlabel('Time')
    grafico2.legend(loc='upper center', ncol=1, fancybox=True, shadow=True)
    grafico2.axis([X_axis[0], X_axis[1], Y_axis_error[0], 0.06])
    # plt.xticks(X_intervals, rotation=45)

    #showing graphic
    plt.show()
    
    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder')):
        os.makedirs( config.get('vis','vis_path_folder'))
    plt.savefig(config.get('vis','vis_path_folder') + '/MAE.png')

def plotGraphShobit(X_train, df1, train_predict, test_predict, scaler):
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

def plot_difference(model1_predict, model2_predict):
    size = model1_predict.size()
    x_axis = np.arange(size)
    plt.plot(x_axis, model1_predict, 'r-', label='without_drift')
    plt.plot(x_axis, model2_predict, 'b-', label='with_drift')
    plt.title("Model Comparison")
    # putting caption and defining the graphic axes
    plt.ylabel('Observations')
    plt.xlabel('Index')
    plt.legend()
    #showing graphic
    plt.show()
    
    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder')):
        os.makedirs( config.get('vis','vis_path_folder'))
    plt.savefig(config.get('vis','vis_path_folder') + '/performance_comparison.png')

def graphMovingAverage(df):
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
