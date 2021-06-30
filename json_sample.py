import json
import os
import pickle
import configparser as cp
import numpy as np
import pandas as pd
config=cp.RawConfigParser()
config.read('config/config.properties')

dirname = config.get('json_result','json_result_path_folder')
if not os.path.exists(dirname):
	os.makedirs(dirname)

json_report = dict()
sep = '/'

def preprocessing_stage():
    # loading all meta info
    # with open(sep.join([config.get('save_path', 'meta_data_save_path'),
    #                     'meta_info_data_diagnostic.pkl']), 'rb') as f:
    #     data_diagnostic_meta_info = pickle.load(f)
    mean_dist = {}
    if config.getboolean('vis', 'mean_plot'):
        vis_path = sep.join([config.get('vis', 'vis_path_folder1'),
                                     'mean_plot.png'])
        d1= {'img_path': vis_path[10:]}
        mean_dist.update(d1)
   

    json_report["Data Diagnostic"] = {
    'mean_dist':mean_dist
        # 'Train Validation Test': {
        #     'Train Validation Test Size': data_diagnostic_meta_info['train_val_test_split'],
        # }
    }

    return None


def modelling_stage():
    # loading all meta info
    # with open(sep.join([config.get('save_path', 'meta_data_save_path'),
    #                     'meta_info_data_diagnostic.pkl']), 'rb') as f:
    #     data_diagnostic_meta_info = pickle.load(f)
    model_performance = {}
    if config.getboolean('vis', 'model_performance'):
        mdl_prm = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_performance.png'])
        d1= {'img_path': mdl_prm[10:]}
        model_performance.update(d1)
        
    model_predection = {}
    if config.getboolean('vis', 'model_predection'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_predection.png'])
        d1= {'img_path': mdl_pred[10:]}
        model_predection.update(d1)

    model_with_drift_detection_performance = {}
    if config.getboolean('vis', 'model_with_drift_detection_performance'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'model_performance_CD.png'])
        d1= {'img_path': mdl_pred[10:]}
        model_with_drift_detection_performance.update(d1)

    model_with_drift_detection_MAE = {}
    if config.getboolean('vis', 'model_with_drift_detection_MAE'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'MAE.png'])
        d1= {'img_path': mdl_pred[10:]}
        model_with_drift_detection_MAE.update(d1)

    models_comparison = {}
    if config.getboolean('vis', 'models_comparison'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'performance_comparison.png'])
        d1= {'img_path': mdl_pred[10:]}
        models_comparison.update(d1)

    models_MAE_comparison = {}
    if config.getboolean('vis', 'models_MAE_comparison'):
        mdl_pred = sep.join([config.get('vis', 'vis_path_folder2'),
                                     'performance_difference_comparison.png'])
        d1= {'img_path': mdl_pred[10:]}
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


def json_dump_generation():
    # Storing meta data into dict
    #if parser.getboolean('loading_data_diagnostic', 'set_execution'):
    preprocessing_stage()
    modelling_stage()
    # dumping the JSON file
    print('json_report', json_report)
    path = os.getcwd() + '/Results'
    # print(path)
    os.chdir(path)
    with open(f'json_metadata.json', 'w') as fp:
        json.dump(json_report, fp)
        print(f'JSON files are dumped')


json_dump_generation()
