from keras.optimizers import SGD
from keras.models import model_from_json
import keras.backend as K
from keras.models import Model
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics
import numpy as np
#from doc2vec2 import main1
from reading_tweets import main1

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

x2,y2 = main1('Tweets2')

feature_model = Model(loaded_model.input,loaded_model.get_layer('encoder_%d' % (int(len(loaded_model.layers) / 2) - 1)).output)
features = feature_model.predict(np.array(x2))


#print('x2', x2[0])

np.savetxt('train.csv', features, delimiter=',')
np.savetxt('train1.csv', np.array(y2), delimiter=',')




import numpy as np
from sklearn import svm, datasets
import pandas as pd
import random
from sklearn.metrics import accuracy_score


print('svm')
clf = svm.SVC(gamma='scale')
clf.fit(x2, y2)  
pred = clf.predict(x2)
print(accuracy_score(y2, pred))

print('svm')
clf = svm.SVC(gamma='scale')
clf.fit(features, y2)  
pred = clf.predict(features)
print(accuracy_score(y2, pred))

'''
features1 = feature_model.predict(np.array(x1))
np.savetxt('test.csv', features1, delimiter=',')
np.savetxt('test1.csv', np.array(y1), delimiter=',')



#print(features[0])

y_pred1 = clf.predict(features1)
print(accuracy_score(y1,y_pred1))




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

'''


