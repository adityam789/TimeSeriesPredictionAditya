import configparser as cp
import pickle
import matplotlib.pyplot as plt
import os

config=cp.RawConfigParser()
config.read('config/config.properties')

with open(config.get('data_path', 'shapValues'), 'rb') as f:
	attributions_ig = pickle.load(f)	
with open(config.get('data_path', 'gradValues'), 'rb') as f:
	attributions_sv = pickle.load(f)	

print(attributions_ig.shape)
print(attributions_sv.shape)

plt.plot(attributions_ig[0],'r')
# plt.plot(attributions_ig[2],'b')
plt.plot(attributions_sv[0],'y')

#saving graphic
if not os.path.exists(config.get('vis','vis_path_folder3')):
    os.makedirs( config.get('vis','vis_path_folder3'))
plt.savefig(config.get('vis','vis_path_folder3') + '/deep_explain.png')

plt.show()