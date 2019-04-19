from keras.models import model_from_json
from time import time
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
#import metrics
from numpy import array
from sklearn.model_selection import KFold
from doc2vec2 import main1
#from NN_model import NN_m1
import numpy as np
from sklearn import svm, datasets
import pandas as pd
import random
from sklearn.metrics import accuracy_score
#from reading_tweets import main1
from fetching_tweets import  main1
#from Reading_mails import main1
import matplotlib.pyplot as plt


def autoencoder(hidden_num, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric auto-encoder model.
    Input:
        hidden_num: list of number of units in each layer of encoder. 
    """
    num_layers = len(hidden_num) - 1
    x = Input(shape=(hidden_num[0],), name='input')
    h = x

    # hidden layers in encoder
    for i in range(num_layers-1):
        h = Dense(hidden_num[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # Deep representation layer
    h = Dense(hidden_num[-1], kernel_initializer=init, name='encoder_%d' % (num_layers - 1))(h)

    y = h
    # hidden layers in decoder
    for i in range(num_layers-1, 0, -1):
        y = Dense(hidden_num[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(hidden_num[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')





def find_results(feat,Y2, features1,y1):
	
	

	c = list(zip(feat,Y2))
	#print('c',c)
	random.shuffle(c)

	features, y2 = zip(*c)
	



	print('DT')
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	clf.fit(features, y2)  
	pred = clf.predict(features)
	print(accuracy_score(y2, pred))
	y_pred1 = clf.predict(features1)
	print(accuracy_score(y1,y_pred1))
	
	
	print('KNN')
	from sklearn.neighbors import KNeighborsClassifier
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(features, y2)  
	pred = clf.predict(features)
	print(accuracy_score(y2, pred))
	y_pred1 = clf.predict(features1)
	print(accuracy_score(y1,y_pred1))
	
	
	print("adaboost")

	clf =AdaBoostClassifier()
	clf.fit(features, y2)  
	pred = clf.predict(features)
	print(accuracy_score(y2, pred))
	y_pred1 = clf.predict(features1)
	print(accuracy_score(y1,y_pred1))
	

	print("mlp")
	from sklearn.neural_network import MLPClassifier
	clf= MLPClassifier(alpha=1)
	clf.fit(features, y2)  
	pred = clf.predict(features)
	print(accuracy_score(y2, pred))
	y_pred1 = clf.predict(features1)
	print(accuracy_score(y1,y_pred1))
	
	print("RF")

	from sklearn.ensemble import RandomForestClassifier

	clf =RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
	clf.fit(features, y2)  
	pred = clf.predict(features)
	print(accuracy_score(y2, pred))
	y_pred1 = clf.predict(features1)
	print(accuracy_score(y1,y_pred1))
	
def NEW_fe(x2,y2):
	
	from keras.models import model_from_json
	json_file = open('results/ae_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("results/ae_weights.h5")
	print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model.compile(optimizer= 'sgd', loss='mse')
	#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	#x2,y2,x1,y1= main1('C501/C50train', 'C501/C50test')

	feature_model = Model(loaded_model.input,loaded_model.get_layer('encoder_%d' % (int(len(loaded_model.layers) / 2) - 1)).output)
	features = feature_model.predict(np.array(x2))
	return features

	

def changed(x2,y2,x1,y1):
	
	from keras.models import model_from_json
	json_file = open('results/ae_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("results/ae_weights.h5")
	print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model.compile(optimizer= 'sgd', loss='mse')
	#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	#x2,y2,x1,y1= main1('C501/C50train', 'C501/C50test')

	feature_model = Model(loaded_model.input,loaded_model.get_layer('encoder_%d' % (int(len(loaded_model.layers) / 2) - 1)).output)
	features = feature_model.predict(np.array(x2))


	#print('x2', x2[0])
	#print('x1', x1[0])
	
	features1 = feature_model.predict(np.array(x1))
	#np.savetxt('test.csv', features1, delimiter=',')
	#np.savetxt('test1.csv', np.array(y1), delimiter=',')

	
	K_fold_data(features, y2)
	
	
import numpy as np
from sklearn.model_selection import KFold 

def K_fold_data(x1,y1):
	#find_results(x1,y1,x1,y1)
	kf = KFold(n_splits=4) # Define the split - into 2 folds 
	kf.get_n_splits(x1) # returns the number of splitting iterations in the cross-validator

	print(kf) 
	for train_index, test_index in kf.split(x1):
		print('TRAIN:', train_index, 'TEST:', test_index)
		X_train=[]
		y_train=[]
		X_test=[]
		y_test=[]
		for tt in train_index:
			X_train.append(x1[tt])
			y_train.append(y1[tt])
		for tt in test_index:
			X_test.append(x1[tt])
			y_test.append(y1[tt])
		
		find_results(X_train, y_train, X_test, y_test)
			


		
if __name__ == "__main__":
    # load dataset
    #x, y = load_data(args.dataset)
    #x2,y2,x1,y1= main1('C501/C50train', 'C501/C50test')
    x2,y2, x1, y1 = main1('Small_tweets/Train', 'Small_tweets/Train')
    #x2,y2 = main1('Final_enron')
    print('len of x2', len(x2))
    x=np.array(x2)
    y= np.array(y2)
    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    # setting parameters
    



    autoencoder, encoder = autoencoder(hidden_num=[x.shape[-1], 100, 100, 500,  10], init=init)
    autoencoder.compile(optimizer= 'adam', loss='mse', metrics=['accuracy'])
    history = autoencoder.fit(x, x, batch_size=128, epochs=350, validation_split=0.33)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accurcay')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    save_dir='results'
    #print('Pretraining time: %ds' % round(time() - t0))
    dirs = save_dir + '/ae_weights.h5'
    dirs1 = save_dir + '/ae_model.json'
    autoencoder.save_weights(dirs)
    model_json = autoencoder.to_json()
    with open(dirs1, "w") as json_file:
         json_file.write(model_json)
    print("Saved model to disk")
    print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
    
    
    changed(x2,y2,x2,y2)
    
    
