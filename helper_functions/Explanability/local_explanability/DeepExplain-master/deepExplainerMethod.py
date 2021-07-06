import configparser as cp
import pickle
from keras import backend as K
from keras.models import Model
# from .examples.utils import plot, plt
from deepexplain.tensorflow import DeepExplain

from keras.models import load_model
model = load_model('Results\model_paths\stock_lstm_model.h5')

config=cp.RawConfigParser()
config.read('config/config.properties')

with open( config.get('data_path', 'train_x'), 'rb') as f:
	X_train=pickle.load(f)
with open( config.get('data_path', 'train_y'), 'rb') as f:
	X_test=pickle.load( f)
with open(config.get('data_path', 'test_y'), 'rb') as f:
    y_test=pickle.load(f)

with DeepExplain(session=K.get_session()) as de:
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = model.layers[0].input
    
    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
    target_tensor = fModel(input_tensor)
    
    xs = X_test[0:10]
    print(xs.shape)

    # attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    #attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs)
    #attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
    #attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    #attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)
    
    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, samples=100)

print(type(attributions_ig))
print(type(attributions_sv))

with open(config.get('data_path', 'shapValues'), 'wb') as f:
	pickle.dump(attributions_ig, f)	
with open(config.get('data_path', 'gradValues'), 'wb') as f:
	pickle.dump(attributions_sv, f)	

# n_cols = 6
# n_rows = int(len(attributions_gradin) / 2)
# fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

# for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_sv)):
#     row, col = divmod(i, 2)
#     plot(xs[i].reshape(28, 28), cmap='Greys', axis=axes[row, col*3]).set_title('Original')
#     plot(a1.reshape(28,28), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
#     plot(a2.reshape(28,28), xi = xs[i], axis=axes[row,col*3+2]).set_title('Shapley Values')