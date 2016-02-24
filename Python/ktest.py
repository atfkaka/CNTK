import sys
sys.path.insert(0, r"e:\keras")
print(sys.path)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
import numpy as np

X_train = np.array(list(zip(range(10), range(10))))
y_train = np.hstack([np.zeros(5), np.ones(5)])
print(X_train)
print(y_train)
ins = [X_train] +[y_train,]
print(ins)
model = Sequential()
model.add(Dense(output_dim=1, input_dim=2, init='uniform'))
model.add(Dense(output_dim=1, input_dim=2, init='uniform'))
    #model.add(Dense(output_dim=1, input_dim=10, init='uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, init='uniform'))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True)
model.predict(ins)