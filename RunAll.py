from helper_functions import structured_pipepline_functions as structured_pipeline 
from configparser import ConfigParser
import shutil

# Source path
source = "C:/Aditya_folders/aditya_personal/TestAing/ML/Pipeline-20210623T171455Z-001/TimeSeriesPredictionAditya/Results"
# Destination path
def h(i):
    destination = "C:/Aditya_folders/aditya_personal/TestAing/ML/Pipeline-20210623T171455Z-001/final"+str(i)
    return destination

def main_script():

    parser = ConfigParser()
    parser.read('config/config.properties')

    k = 0

    for i in ['./datasets/AAPL1.csv', './datasets/aadr.us.txt', './datasets/acwx.us.txt', './datasets/airr.us.txt', './datasets/AAPL.csv']:
        print('Hi')
        if parser.get('FLAGS', 'execute.preprocessing').lower() == 'true':
            try:
                print("\nStarting Preprocessing\n") 
                structured_pipeline.pre_processing(i) 
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
        # Move the content of
        # source to destination
        k += 1
        dest = shutil.move(source, h(k), copy_function = shutil.copytree)
        print(dest)

if (__name__=='__main__'):
    main_script()
