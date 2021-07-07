import json
import os
import configparser
from configparser import ConfigParser
import shutil

parser = ConfigParser()
parser.read('./config/config.properties')
json_dump = dict()
sep = "/"

def readpngfiles(path):
  files = []
  # r=root, d=directories, f = files
  for r, d, f in os.walk(path):
    for file in f:
      if '.png' in file:
        files.append(os.path.join(r, file).replace("\\","/"))
  print(files)
  return files

def preprocess():
  with open(sep.join(['data_paths', 'data13.json']),'rb') as path:
    data_head10 = json.load(path)
  with open(sep.join(['data_paths', 'data_stats.json']),'rb') as f:
    data_stats = json.load(f)
  json_dump["Data Preprocessing & Modelling"] = { 
    "Top 10 Columns of data": data_head10,
    "Data Statistics": data_stats,
    "Correlation Matrix": [sep.join(["data_paths", "correlation_table.png"])],
    "Confusion Matrix": [sep.join(["model_paths","confusion_matrix.png"])],
    "Class Distribution" : [sep.join(["data_paths", "class_distribution.png"])]
  }
 

def Global_Explainability():
  one_dts = "Decision tree importance is calculated based on surrogate split ,information gain.Here surrogate tree tries to predict actual split another decision tree is created to predict your split (based on weights)."
  one_rf = "Feature Importance is calculated using Gini Importance(mean decrease in impurity).Mean decrease in accuracy happens when features are correlated."
  one_shap_global = "In this technique we find importance of particular feature it subsets the different features other than X and find the difference in predicted values and averaged so as to obtain the shapley values.Global Importance is average of shapley values arranged in descending order."
  one_PDP = "PDP plots are prediction probabilities of classification against changes in the value of feature.The values are means of several test cases."
  one_gis = "Feature score for each column after calculating combined score of all global explainability techniques" 
  append_str = './Global_Visualization/PDP'
  arr = parser.get('Global_visualization','pdp')
  #print(arr)
  test_list = arr
  pdp_list = [append_str + sub for sub in test_list]
  #print(pdp_list)
  json_dump["Global Explainability"] = {
	    "Random Forest Model (RF)":{
	        "one_liner": one_rf,
	        "img_path": [sep.join(["Global_Visualization","Random_Forest","Random_forest_feature_importance.png"])]
	    },
	    "Decision Tree Surrogates(DTS)": {
	        "one_liner": one_dts,
	        "img_path": [sep.join(["Global_Visualization","DTS","DTS_feature_importance.png"])]
	    },
	    "Shap Global":{
	        "one_liner": one_shap_global,
	        "img_path": [sep.join(["Global_Visualization","Shap","shap.png"])]
	    },
	    "Partial Dependency Plot(PDP)": {
	        "one_liner": one_PDP,
	        "img_path": pdp_list
	    },
	    "Global Index Score":{
	        "one_liner": one_gis,
	        "img_path": [sep.join(["Global_Visualization","GIS","global_feature_score.png"])]
	    }
	}
  print(json_dump)


def Local_Explainability():
  one_shap_local = "It uses an additive feature attribution so as to explain local feature importance by using shapley values."
  one_ICE = "This Technique displays one line for each instance that shows how instance prediction changes when the feature changes."
  one_loco = "It will find prediction probabilities of test data then remove one of the feature from the dataset and retrain the data , find the prediction probabilities again without removed feature.Take the difference between initial and new prediction probabilities and continue the process for all the features."
  one_lime = "It creates fake data for each observation and then computes the similarity score between original and fake data.It will chose ‘m’ features along with similarity scores and create simple model.Feature weights from the simple model makes explanations for complex model local behaviour."
  append_str0 = "Local_Visualization/Shap/"
  shap_list = readpngfiles(append_str0)
  #print(shap_list)
  arr1 = os.listdir(parser.get('Local_visualization','ICE'))
  append_str1 = "Local_Visualization/ICE/"
  test_list = arr1
  ice_list = [append_str1 + sub for sub in test_list]
  #print(ice_list)
  append_str2 = "Local_Visualization/Loco/"
  loco_list = readpngfiles(append_str2)
  #print(loco_list)
  append_str3 = "Local_Visualization/Lime/"
  lime_list = readpngfiles(append_str3)
  #print(lime_list)
  json_dump["Local Explainability"] = {
      "Shap Local":{
          "one_liner": one_shap_local,
          "img_path":shap_list,
      },
      "Individual Conditional Expectation(ICE)": {
          "one_liner": one_ICE,
          "img_path":ice_list
      },
      "Leave One Covariate Out(LOCO)":{
          "one_liner": one_loco,
          "img_path":loco_list,
      },
      "Local Interpretable Model Agnostic Explanations(LIME)":{
          "one_liner": one_lime,
          "img_path":lime_list,          
      }      
  }
  print(json_dump)

def Bias():
  fairml_one_line = "Fairml helps to determine the relative significance of the inputs to a black-box predictive model in order to assess the model’s fairness.It leverages model compression and four input ranking algorithms to quantify a model’s relative predictive dependence on its inputs.The basic idea behind FairML (and many other attempts to audit or interpret model behavior) is to measure a model’s dependence on its inputs by changing them."
  bias_pu_one_line = "Bias postive labelled problem occurs when most of our data belongs to one particular class or only a small part of our data is labelled then we use this technique."
  bias_multimodel_one_line = "In Bias Multimodel we determine the feature subset that we think may define our bias and classify records as belonging to biased set or not using different models but with same parameters.Here we make an intial classifier such that it can capture that bias or at least approximate it with small subset of available fatures."
  bias_one_line = "Sensitive attributes dealing with bias"

  with open(sep.join(['Bias', 'bias_postive_labelled.json']), 'rb') as f:
    bias_postive_labelled  = json.load(f)
  with open(sep.join(['Bias', 'bias_multimodel.json']), 'rb') as f:
    bias_multimodel = json.load(f)
  with open(sep.join(['Bias', 'bias_dict.json']), 'rb') as f:
    bias_dict = json.load(f)
  
  json_dump["Bias Techniques"] = {
      "FairML":{
          "one_liner": fairml_one_line,
          "img_path":[sep.join(["Bias","fairml.png"])]
      },

      "Sensitive Attributes":{
          "one_liner": bias_one_line,
          "Bias variables": bias_dict
      },

      "Bias Postive Unlabelled": {
          "one_liner": bias_pu_one_line,
          "Bias PU": bias_postive_labelled
      },

      "Bias Multi-Model Feature Set Segmentation": {
          "one_liner": bias_multimodel_one_line,
          "Bias Multimodel": bias_multimodel
      }
  }
  print(json_dump)

def ART():
  with open(sep.join(["ART", "ART_data.json"]),'rb') as path:
    ART_data = json.load(path)
  with open(sep.join(["ART", "query_data.json"]),'rb') as path1:
    query_data = json.load(path1)

  one_ART = "ART helps us create the adversarial samples by adding random noise to the samples, which are kind of corner cases which are used by attackers to fool the model. This gives us a kind of hint about where our model would fail."
  json_dump["ART"] = {
      "one_liner": one_ART,
      "query_data": query_data,
      "Adversarial Data": ART_data,
      "ART plots": sep.join(["ART","scores.png"])
 }
  print(json_dump)


def Synthetic_data():
  with open(sep.join(["synthetic_data", "synthetic_data.json"]),'rb') as path:
    data_sdv = json.load(path)
  one_sdv = "Synthetic data generated from training data using GAN's"
  json_dump["Synthetic Data"] = {
      "one_liner": one_sdv,
      "Synthetic Data": data_sdv
 }
  print(json_dump)

def counterfactual_gen():
  with open(sep.join(["counterfactuals", "diverse_counterfactuals.json"]),'rb') as path:
    diverse_counterfactual = json.load(path)
  with open(sep.join(["counterfactuals", "query_instance.json"]),'rb') as path:
    query_instances = json.load(path)

  one_counterfactual = "*******************************************************"
  json_dump["Counterfactuals Generated"] = {
      "one_liner": one_counterfactual,
      "Query Instance": query_instances,
      "Diverse Counterfactuals": diverse_counterfactual
 }
  print(json_dump)



def json_drop(data):
  original_path = os.getcwd()
  os.chdir(original_path)
  with open(f'./json_metadata.json', 'w') as fp:
    json.dump(data, fp)
    print(f'JSON files are dumped')


def json_generator():
  original_path = os.getcwd()
  path_results = os.getcwd() + '/results'
  if not os.path.isdir(path_results):
     os.makedirs(path_results)
  move_list = ['data_paths','model_paths']
  for file in move_list:
    shutil.move(file,path_results)
  os.chdir(path_results)
   
  preprocess()
  
  os.chdir(original_path)
  config = configparser.ConfigParser()
  config.read('config/master_config.properties')
  
  
  if config.get('FLAGS', 'execute.global_explainability').lower()=='true':
    shutil.move('Global_Visualization',path_results)
    os.chdir(path_results)
    Global_Explainability()

  if config.get('FLAGS', 'execute.local_explainability').lower()=='true':
    os.chdir(original_path)
    shutil.move('Local_Visualization',path_results)
    os.chdir(path_results)
    Local_Explainability()

  if config.get('FLAGS', 'execute.bias').lower()=='true' or config.get('FLAGS', 'execute.fair_ml').lower()=='true':
    os.chdir(original_path)
    shutil.move('Bias',path_results)
    os.chdir(path_results)
    Bias()

  if config.get('FLAGS', 'execute.art').lower()=='true':
    os.chdir(original_path)
    shutil.move('ART',path_results)
    os.chdir(path_results)
    ART()

  if config.get('FLAGS', 'execute.synthetic_data_generator').lower()=='true':
    os.chdir(original_path)
    shutil.move('synthetic_data',path_results)
    os.chdir(path_results)
    Synthetic_data()

  if config.get('FLAGS', 'execute.counterfactual_generator').lower()=='true':
    os.chdir(original_path)
    shutil.move('counterfactuals',path_results)
    os.chdir(path_results)
    counterfactual_gen()
  # Dumping JSON data
  json_drop(json_dump)