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
    detector_width = 1
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
    X_intervals = range(X_axis[0], X_axis[1], 900)

    label_detection_found = 'Drift Found'

    desc = "This is a plot showcasing the trends of the dataset and the forecast/ prediction of the model utilizing detection of drifts. The dashed lines represent the indexes of the dataset where drift was found. This is to give an overview of the dataset and model. \n The x-axis represents the indexes of data in the input and the y-axis represents the name of the dataset"

    # Creating a figure
    figure = plt.figure()
    if(name != None):
        figure.suptitle(name, fontsize=11, fontweight='bold')

    # Defining Plot 1 with preview and actual data
    graph1 = figure.add_subplot()
    graph1.subplots_adjust(top=0.85)
    # plt.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    graph1.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
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
    plt.ylabel(config.get('graph_labels','data_type'))
    plt.xlabel('Index')
    # graph1.axis([X_axis[0], X_axis[1], 0, 1])
    graph1.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    # plt.xticks(X_intervals, rotation=45)

    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder2')):
        os.makedirs( config.get('vis','vis_path_folder2'))
    plt.savefig(config.get('vis','vis_path_folder2') + '/model_performance_CD.png')

    #showing graphic
    plt.show()

    plt.close()


def plotGraphError(stream, detections, error_stream_vector, n, name):

    error_color = 'Blue'
    detector_width = 1
    style = 'dashed'
    label_detection_found = 'Drift Found'

    detection_found_color = '#00BFFF'
    alpha_retraining_color = 0.15
    retraining_color = 'Gray'

    X_axis = [0] * 2
    X_axis[0] = 0
    X_axis[1] = len(stream)
    X_intervals = range(X_axis[0], X_axis[1], 900)

    desc = "This is a graph displaying the forecasting error of the concept drift aware model. The forecasting error is the absolute error(AE) between the prediction/ forecast of the model and the real value(Data). \n The dashed lines are for the indexes at which drift was detected. \n The x-axis represents the indexes of the dataset and the y-axis represents the absolute error or difference"

    # Creating a figure
    figure2 = plt.figure()
    if(name != None):
        figure2.suptitle(name, fontsize=11, fontweight='bold')

    # Defining Plot 2 with the forbearing error
    grafico2 = figure2.add_subplot()
    grafico2.subplots_adjust(top=0.85)
    # grafico2.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    grafico2.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    grafico2.plot(error_stream_vector,
                  label='Forecasting Error (Between Model and Real)', color=error_color, linewidth=0.5)
    grafico2.set_title("Forecasting Error (Between Model and Real)")

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
    plt.ylabel('AE')
    plt.xlabel('Index')
    grafico2.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    # grafico2.axis([X_axis[0], X_axis[1], Y_axis_error[0], 0.06])
    # plt.xticks(X_intervals, rotation=45)
    
    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder2')):
        os.makedirs( config.get('vis','vis_path_folder2'))
    plt.savefig(config.get('vis','vis_path_folder2') + '/MAE.png')

    #showing graphic
    plt.show()

def plotGraphShobit(X_train, df1, train_predict, test_predict, scaler):
    desc = "This is graph illustrating the trend of the dataset and the forecast/ prediction of the model without detection of drifts. \n The model utilizes training and test data (split part of dataset). \n The red line is the forecast of the training data and the yellow line is the forecast of the test data. This is to give an overview of the dataset and model. \n The x-axis represents the indexes of data in the input and the y-axis represents the name of the dataset"
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
    plt.title('Real dataset and forecast')
    plt.suptitle(config.get('graph_labels','data_title'))
    plt.xlabel('Index', fontsize=18)
    plt.ylabel(config.get('graph_labels','data_type'), fontsize=18)
    plt.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
	# plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1),label='True')
    plt.plot(trainPredictPlot,label='Train')
    plt.plot(testPredictPlot,label='Test')
    plt.legend()		
    if not os.path.exists(config.get('vis','vis_path_folder2')):
    	os.makedirs( config.get('vis','vis_path_folder2'))
    plt.savefig(config.get('vis','vis_path_folder2') + '/model_performance.png')

def plot_difference(model1_predict, model2_predict, real):
    desc = "This is a graph ploting the results/ forecast of both the models and the dataset. \n This is to provide a bigger picture of the models performance/ forecast and the real data. \n The x-axis represents the indexes of the data and the y-axis represents the data name."
    size = model1_predict.size
    x_axis = np.arange(size)
    plt.plot(x_axis, model1_predict, 'r-', label='without_drift', linewidth=0.5)
    plt.plot(x_axis, model2_predict, 'b-', label='with_drift', linewidth=0.5)
    plt.plot(x_axis, real, 'g-', label='Real Values', linewidth=0.5)
    plt.suptitle(config.get('graph_labels','data_title'))
    plt.title("Model Forecast Comparison")
    # plt.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    # putting caption and defining the graphic axes
    plt.ylabel(config.get('graph_labels','data_type'))
    plt.xlabel('Index')
    plt.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    
    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder2')):
        os.makedirs( config.get('vis','vis_path_folder2'))
    plt.savefig(config.get('vis','vis_path_folder2') + '/performance_comparison.png')

    #showing graphic
    plt.show()

    plt.close()

def plot_difference_comparison(model1_predict_MAE, model2_predict_MAE):

    desc = "This is plot showing the forecasting error of both the models (one with concept drift aware system and one without). \n This is to compare the performance/ accuracy of the models. \n The forecasting error is the absolute difference between the model prediction and the real data. \n The x-axis represents the indexes of the data and the y-axis represents the AE (Absolute error)"
    size = len(model1_predict_MAE)
    x_axis = np.arange(size)
    plt.plot(x_axis, model1_predict_MAE, 'r-', label='without_drift', linewidth=0.5)
    plt.plot(x_axis, model2_predict_MAE, 'b-', label='with_drift', linewidth=0.5)
    plt.suptitle(config.get('graph_labels','data_title'))
    plt.title("Models AE Comparison")
    # putting caption and defining the graphic axes
    plt.ylabel('AE')
    plt.xlabel('Index')
    plt.legend()
    # plt.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    #saving graphic
    if not os.path.exists(config.get('vis','vis_path_folder2')):
        os.makedirs( config.get('vis','vis_path_folder2'))
    plt.savefig(config.get('vis','vis_path_folder2') + '/performance_difference_comparison.png')

    #showing graphic
    plt.show()

    plt.close()

def graphMovingAverage(df):

    desc = "This is a graph for visualizing the movement of the dataset. \n It uses a simple fundamental math concept of moving average of size 10 and 20 to show the general trend of data. \n The x-axis represents the indexes of the data and the y-axis represents the data name."

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
    # plt.figtext(0.5, 0.01, "one text and next text", ha="center", fontsize=18, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.annotate(desc, (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.plot(df[target_column].tail(200), label='df')
    plt.plot(df_10[target_column].tail(200), label='df_10')
    plt.plot(df_20[target_column].tail(200), label='df_20')
    # plt.plot(df_30[target_column].tail(200), label='df_30')
    # plt.plot(df_40[target_column].tail(200), label='df_40')
    plt.suptitle(config.get('graph_labels','data_title'))
    plt.title('Moving average of the data')
    plt.xlabel('Index')
    plt.ylabel(config.get('graph_labels','data_type'))
    plt.legend(loc='upper left')

    if not os.path.exists(config.get('vis', 'vis_path_folder1')):
    	os.makedirs(config.get('vis', 'vis_path_folder1'))
    plt.savefig(config.get('vis', 'vis_path_folder1') + '/mean_plot.png')
