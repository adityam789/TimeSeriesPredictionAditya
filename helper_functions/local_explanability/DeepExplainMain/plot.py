import configparser as cp
import matplotlib.pyplot as plt
import os

def plotter(i, imgPath, attributions_ig, attributions_sv):

	config=cp.RawConfigParser()
	config.read('config/config.properties')	

	print(attributions_ig.shape)
	print(attributions_sv.shape)

	fig = plt.figure(figsize=(10,5))
	graph = fig.add_subplot()
	graph.plot(attributions_ig[i],'r')
	graph.plot(attributions_sv[i],'y')
	plt.title('Deep explainer')
	plt.xlabel('Days or Features of input')
	plt.ylabel('Attributions')

	#saving graphic
	if not os.path.exists(config.get('vis','vis_path_folder3')):
		os.makedirs( config.get('vis','vis_path_folder3'))
	plt.savefig(config.get('vis','vis_path_folder3') + imgPath)

	# plt.show()