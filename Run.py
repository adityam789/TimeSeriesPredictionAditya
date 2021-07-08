from helper_functions import structured_pipepline_functions as structured_pipeline
from configparser import ConfigParser
from helper_functions.local_explainer import local_explainer_functions
from helper_functions.global_explainer import global_explainer_functions

def main_script():
    parser = ConfigParser()
    parser.read('config/config.properties')
    if parser.get('FLAGS', 'execute.preprocessing').lower() == 'true':
        # try:
        print("\nStarting Preprocessing\n")
        structured_pipeline.pre_processing()
        print("\nPreprocessing stage successful\n")
        # except:
        #     print("\nError in Preprocessing stage\n")
    if parser.get('FLAGS', 'execute.modelling').lower() == 'true':
        # try:
        print("\nStarting Modelling \n")
        structured_pipeline.modelling()
        print("\nModelling stage successful\n")
        # except Exception as e:
        #     print(e)
        #     print("\nError in Modelling stage\n")
    if parser.get('FLAGS', 'execute.localExplainabilty').lower() == 'true':
        # try:
        print("\nStarting local explaination\n")
        local_explainer_functions()
        print("\nlocal explaination stage successful\n")
        # except Exception as e:
        #     print(e)
        #     print("\nError in Modelling stage\n")
    if parser.get('FLAGS', 'execute.globalExplainabilty').lower() == 'true':
        # try:
        print("\nStarting global explaination\n")
        global_explainer_functions()
        print("\global explaination stage successful\n")
        # except Exception as e:
        #     print(e)
        #     print("\nError in Modelling stage\n")
            
if (__name__ == '__main__'):
    main_script()