from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
import numpy as np

#Generating datda
nsamples = 20
X_train = np.array(list(zip(range(nsamples), range(nsamples))))
y_train = np.hstack([np.zeros(nsamples//2), np.ones(nsamples//2)])
ins = [X_train] +[y_train,]

#Build the model
model = Sequential()
model.add(Dense(output_dim=1, input_dim=2, init='uniform'))
model.add(Activation('softmax'))

#Define the optimizer
sgd = SGD(lr=0.2)

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#Train
model.fit(X_train, y_train, nb_epoch=100, batch_size=2)

#Predict
actual = model.predict(ins)
print("Input:")
print(np.hstack([X_train, y_train[np.newaxis].T]))
print("Predicted (class probabilities for class 0 and 1) + prediction")
print(np.hstack([actual, np.array(actual[:,0]<actual[:,1], dtype=int)[np.newaxis].T]))
assert len(actual)==len(X_train)
assert nsamples>=10
assert np.all(actual[:4][:,0] > actual[:4][:,1])
assert np.all(actual[-4:][:,0] < actual[-4:][:,1])

