from helper_functions import structured_pipepline_functions as structured_pipeline 
from configparser import ConfigParser

def main_script():


    parser = ConfigParser()

    parser.read('config/config.properties')

    if parser.get('FLAGS', 'execute.preprocessing').lower() == 'true':

        try:

            print("\nStarting Preprocessing\n") 
            structured_pipeline.pre_processing() 
            print("\nPreprocessing stage successful\n")

        except Exception as e:

            print(e)
            print("\nError in Preprocessing stage\n")

    if parser.get('FLAGS','execute.modelling').lower() =='true':

        try:

            print("\nStarting Modelling \n") 
            structured_pipeline.modelling()
            print("\nModelling stage successful\n")

        except Exception as e:

            print(e)
            print("\nError in Modelling stage\n")



if (__name__=='__main__'):
    main_script()
