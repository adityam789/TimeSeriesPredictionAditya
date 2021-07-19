import json
import os
import configparser as cp

config=cp.RawConfigParser()
config.read('config/config.properties')

dirname = config.get('json_result','json_result_path_folder')
if not os.path.exists(dirname):
	os.makedirs(dirname)

json_report = dict()
sep = '/'

def readpngfiles(path):
  files = []
  # r=root, d=directories, f = files
  for r, d, f in os.walk(path):
    for file in f:
      if '.png' in file:
        files.append(os.path.join(r, file).replace("\\","/"))
#   print(files)
  return files

def preprocessing_stage():

    mean_dist_words = "This is a graph for visualizing the movement of the dataset.\nIt uses a simple fundamental math concept of moving average of size 10 and 20 to show the general trend of data.\nThe x-axis represents the indexes of the data and the y-axis represents the data name."

    mean_dist = {}
    if config.getboolean('vis', 'mean_plot'):
        vis_path = sep.join([config.get('vis', 'vis_path_folder1'),
                                     'mean_plot.png'])
        d1= {
            'img_path': vis_path[10:],
            # 'one_liner': mean_dist_words
        }
        mean_dist.update(d1)
   

    json_report["Data Diagnostic"] = {
    'mean_dist':mean_dist
        # 'Train Validation Test': {
        #     'Train Validation Test Size': data_diagnostic_meta_info['train_val_test_split'],
        # }
    }

    return None


def modelling_stage():

    n_steps = config.getint('features','time_step')+1

    model_performance_words = "This is graph illustrating the trend of the dataset and the forecast/ prediction of the model without detection of drifts. The model utilizes\ntraining and test data (split part of dataset). The red line is the forecast of training data and yellow line is the forecast of test data.\nThis is to give an overview of the dataset and model. The x-axis and y-axis represents the indexes of data and the data name/type"
    model_predection_words = "This is a graph illustrating the future prediction of model.\nThe plot showcases the data of past " + str(n_steps - 1) + " indexes (days) and predicts the future values for the next 10 indexes (days).\nThe x-axis represents the indexes of the data and the y-axis represents the data name/ type."
    model_with_drift_detection_MAE_words = "This is a graph displaying the forecasting error of the concept drift aware model. The forecasting error is the absolute error(AE)\nbetween the prediction/ forecast of the model and the real value(Data). The dashed lines are for the indexes at which drift was\ndetected. The x-axis and y-axis represents the indexes of the dataset and the absolute error/difference"
    model_with_drift_detection_performance_words = "This is a plot showcasing the trends of the dataset and the forecast/ prediction of the model utilizing detection of drifts.\nThe dashed lines represent the indexes of the dataset where drift was found. This is to give an overview of the dataset and model.\nThe x-axis represents the indexes of data in the input and the y-axis represents the name of the dataset"
    models_comparison_words = "This is a graph ploting the results/ forecast of both the models and the dataset.\nThis is to provide a bigger picture of the models performance/ forecast and the real data.\nThe x-axis represents the indexes of the data and the y-axis represents the data name."
    models_MAE_comparison_words = "This is plot showing the forecasting error of both the models (one with concept drift aware system and one without).\nThis is to compare the performance/ accuracy of the models. The forecasting error is the absolute difference between the model\nprediction and real data. The x-axis represents the indexes of the data and the y-axis represents the AE (Absolute error)"

    model_performance = {}
    if config.getboolean('vis', 'model_performance'):
        mdl_prm = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_performance.png'])
        d1= {
            'img_path': mdl_prm[10:],
            # 'one_liner': model_performance_words
        }
        model_performance.update(d1)
        
    model_predection = {}
    if config.getboolean('vis', 'model_predection'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_predection.png'])
        d1= {
            'img_path': mdl_pred[10:],
            # 'one_liner': model_predection_words
        }
        model_predection.update(d1)

    model_with_drift_detection_performance = {}
    if config.getboolean('vis', 'model_with_drift_detection_performance'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_performance_CD.png'])
        d1= {
            'img_path': mdl_pred[10:],
            # 'one_liner': model_with_drift_detection_performance_words
        }
        model_with_drift_detection_performance.update(d1)

    model_with_drift_detection_MAE = {}
    if config.getboolean('vis', 'model_with_drift_detection_MAE'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'MAE.png'])
        d1= {
            'img_path': mdl_pred[10:],
            # 'one_liner': model_with_drift_detection_MAE_words
        }
        model_with_drift_detection_MAE.update(d1)

    models_comparison = {}
    if config.getboolean('vis', 'models_comparison'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'performance_comparison.png'])
        d1= {
            'img_path': mdl_pred[10:],
            # 'one_liner': models_comparison_words
        }
        models_comparison.update(d1)

    models_MAE_comparison = {}
    if config.getboolean('vis', 'models_MAE_comparison'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'performance_difference_comparison.png'])
        d1= {
            'img_path': mdl_pred[10:],
            # 'one_liner': models_MAE_comparison_words
        }
        models_MAE_comparison.update(d1)

    json_report["Modelling"] = {
        'model_performance':model_performance,
        'model_predection':model_predection,
        'model_with_drift_detection_performance':model_with_drift_detection_performance,
        'model_with_drift_detection_MAE':model_with_drift_detection_MAE,
        'models_comparison':models_comparison,
        'models_MAE_comparison':models_MAE_comparison
        # 'Train Validation Test': {
        #     'Train Validation Test Size': data_diagnostic_meta_info['train_val_test_split'],
        # }
    }

    return None

def local_explainabilty_stage(path):

    deep_explain_words = "This shows attribution of every feature (Days in the time step) towards the target. It shows both the attribution by Integrated Gradients and Shapley Value sampling for a particular instance in the test dataset. Attribution is a real value R(x_i) for each input feature, with respect to a target neuron of interest. Positive value of feature shows that it contribute positively to the activation of the target output and vice-versa"
    source_citation = ["https://github.com/marcoancona/DeepExplain", "arxiv.org/abs/1711.06104"]
    about = "A unified framework of perturbation and gradient-based attribution methods for Deep Neural Networks interpretability. DeepExplain also includes support for Shapley Values sampling. (ICLR 2018)"
    # lime_explain_words = "This shows the local explanation of a particular instance in the training set. The graph shows the explanation of all 100 features for an instance. A local explanation is a local linear approximation of the model's behaviour around the vicinity of a particular instance."
    lime_explain_minmax_words = "This shows the 10 highest weighted features/ days and 10 least weighted features of a particular instance while generating its local explanability. A local explanation is a local linear approximation of the model's behaviour around the vicinity of a particular instance."
    source_citation = ["https://arxiv.org/abs/1602.04938", "https://github.com/marcotcr/lime", "https://stackoverflow.com/questions/61511874/how-can-i-use-lime-to-classify-my-time-series"]
    about = "Lime: Explaining the predictions of any machine learning classifier"

    file_dir = readpngfiles(path+'/local_exp_paths')
    lime_explain_img_highest_error = []
    lime_explain_img_lowest_error = []
    deep_explain_img_highest_error = []
    deep_explain_img_lowest_error = []

    for i in file_dir:
        index = i.find('local_exp_paths/lime_explain')
        if index != -1:
            subIndex = i[index:].find("/max")
            if subIndex == -1:
                lime_explain_img_highest_error.append(i[index:])
            else:
                lime_explain_img_lowest_error.append(i[index:])
        else:
            index = i.find('local_exp_paths/deep_explain')
            if index != -1:
                subIndex = i[index:].find("/max")
                if subIndex == -1:
                    deep_explain_img_highest_error.append(i[index:])
                else:
                    deep_explain_img_lowest_error.append(i[index:])
            else:
                print("Error")
    
    deep_explain = {}
    if config.getboolean('vis', 'deep_explain'):
        mdl_prm = sep.join([config.get('vis', 'vis_path_folder3'),
                                     'deep_explain.png'])
        d1= {
            'one_liner': deep_explain_words,
            'highest error instances':{
                'img_path': deep_explain_img_highest_error
            },
            'least error instances':{
                'img_path': deep_explain_img_lowest_error
            }
        }
        deep_explain.update(d1)
        
    # lime_explain = {}
    # if config.getboolean('vis', 'lime_explain'):
    #     mdl_pred = sep.join([config.get('vis', 'vis_path_folder3'),
    #                                  'lime_explain.png'])
    #     d1= {
    #         'one_liner': lime_explain_words,
    #         'img_path': mdl_pred[10:]
    #     }
    #     lime_explain.update(d1)

    lime_explain_minmax = {}
    if config.getboolean('vis', 'lime_explain_minmax'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder4'),
                                     'lime_explain_minmax.png'])
        d1= {
            'one_liner': lime_explain_minmax_words,
            'highest error instances':{
                'img_path': lime_explain_img_highest_error
            },
            'least error instances':{
                'img_path': lime_explain_img_lowest_error
            }
        }
        lime_explain_minmax.update(d1)

    log_return_explain = {}
    if config.getboolean('vis', 'long_return_explain'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder7'),
                                     'long_return_explain.png'])
        d1= {
            'one_liner': "",
            'img_path': mdl_pred[10:]
            # 'highest error instances':{
            #     'img_path': lime_explain_img_highest_error
            # },
            # 'least error instances':{
            #     'img_path': lime_explain_img_lowest_error
            # }
        }
        log_return_explain.update(d1)

    json_report["local_explainabilty"] = {
        'deep_explain':deep_explain,
        'lime_explain_minmax':lime_explain_minmax,
        'log_return_explain':log_return_explain
    }

    return None

def global_explainabilty_stage():

    xai_explain_words = "This is a plot showing the importance of each feature/ day in the time step by showing the loss created upon randomly shuffling any one feature/day. More the negative loss, More important the feature is. It uses the test dataset to create loss evaluations."
    source_citation = ["https://github.com/EthicalML/xai", "https://ethical.institute/principles.html#commitment-3", "https://ethicalml.github.io/xai/index.html"]
    videos = ["https://www.youtube.com/watch?v=vq8mDiDODhc", "https://www.youtube.com/watch?v=GZpfBhQJ0H4"]
    about = "XAI - An eXplainability toolbox for machine learning"

    xai_explain = {}
    if config.getboolean('vis', 'xai_explain'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder5'),
                                     'xai_explain.png'])
        d1= {
            'one_liner': xai_explain_words,
            'img_path': mdl_pred[10:]
        }
        xai_explain.update(d1)

    ALE_explain = {}
    if config.getboolean('vis', 'ALE_explain'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder6'),
                                     'ALE_explain_max5_features.png'])
        mdl_pred2 = sep.join([config.get('vis', 'vis_path_folder6'),
                                     'ALE_explain_min5_features.png'])
        d1= {
            'one_liner': "",
            'img_path': [mdl_pred[10:], mdl_pred2[10:]]
        }
        ALE_explain.update(d1)

    SHAP_global_explain = {}
    if config.getboolean('vis', 'shap_global_explain'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder8'),
                                     'shap_global_explain_summary_plot.png'])
        mdl_pred2 = sep.join([config.get('vis', 'vis_path_folder8'),
                                     'shap_global_explain_force_plot.png'])
        d1= {
            'one_liner': "",
            'img_path': [mdl_pred[10:], mdl_pred2[10:]]
        }
        ALE_explain.update(d1)

    json_report["global_explainabilty"] = {
        'xai_explain':xai_explain,
        'ALE_explain':ALE_explain,
        'SHAP_global_explain':SHAP_global_explain
    }

    return None

def json_dump_generation():
    # Storing meta data into dict
    #if parser.getboolean('loading_data_diagnostic', 'set_execution'):
    path = os.getcwd() + '/Results'
    # print(path)
    preprocessing_stage()
    modelling_stage()
    local_explainabilty_stage(path)
    global_explainabilty_stage()
    # dumping the JSON file
    # print('json_report', json_report)
    os.chdir(path)
    with open(f'json_metadata.json', 'w') as fp:
        json.dump(json_report, fp)
        print(f'JSON files are dumped')


json_dump_generation()
