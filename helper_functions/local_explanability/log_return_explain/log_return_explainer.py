import configparser as cp
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd
import os

model = load_model('Results\model_paths\stock_lstm_model.h5')

config = cp.RawConfigParser()
config.read('config/config.properties')
with open(config.get('data_path', 'train_x'), 'rb') as f:
    X_train = pickle.load(f)
with open(config.get('data_path', 'test_x'), 'rb') as f:
    y_train = pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
    X_test=pickle.load( f)

df1=X_train[0]
df_3=model.predict(X_train[1:11].reshape(10,100,1))
df3=pd.DataFrame(df_3,columns=['Close'])
df3["returns"] = df3.Close.pct_change()
df3["log_returns"] = np.log(1 + df3["returns"])
df2=pd.DataFrame(df1,columns=['Close'])
df2["returns"] = df2.Close.pct_change()
df2["log_returns"] = np.log(1 + df2["returns"])
plt.figure(1, figsize=(16, 4))
plt.plot(df2.log_returns)
plt.plot(df3.log_returns)

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder7')):
    os.makedirs( config.get('vis','vis_path_folder7'))
plt.savefig(config.get('vis','vis_path_folder7') + '/log_return_explain.png')