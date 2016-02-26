import sys
sys.path.insert(0, r"e:\keras")
print(sys.path)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
import numpy as np

#Generating datda
X_train = np.array(list(zip(range(10), range(10))))
y_train = np.hstack([np.zeros(5), np.ones(5)])
ins = [X_train] +[y_train,]

#Build the model
model = Sequential()
model.add(Dense(output_dim=1, input_dim=2, init='uniform'))
model.add(Activation('softmax'))

#Define the optimizer
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#Train
model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True)

#Predict
actual = model.predict(ins)
print("Input:")
print(np.hstack([X_train, y_train[np.newaxis].T]))
print("Predicted (class probabilities for class 0 and 1) + prediction")
print(np.hstack([actual, np.array(actual[:,0]<actual[:,1])[np.newaxis].T]))
assert len(actual)==len(X_train)
assert np.all(actual[:4][:,0] > actual[:4][:,1])
assert np.all(actual[-4:][:,0] < actual[-4:][:,1])

