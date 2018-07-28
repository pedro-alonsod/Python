import numpy as np
n = 10
X = np.random.rand(n, 2) * 4 - 2
labels = X[:,0] ** 2 + X[:,1] ** 2 < 1
print(X)
print(labels)

Y = np.zeros((n, 2))
Y[np.arange(n), labels.astype(int)] = 1
print(Y)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap='brg')
plt.show()

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(output_dim=2, input_dim=2))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png', show_shapes=True)

model.fit(X, Y, nb_epoch=50, batch_size=5)

X_valid = np.random.rand(n, 2) * 4 - 2
Y_pred = model.predict(X_valid)
print(Y_pred)

from keras.layers import Activation
model = Sequential()
model.add(Dense(output_dim=3, input_dim=2))
  
model.add(Activation('tanh'))
model.add(Dense(output_dim=3, input_dim=3))
model.add(Activation('tanh'))
model.add(Dense(output_dim=2, input_dim=3))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.save_weights('model.weights')
with open('model.structure', 'w') as fp:
    fp.write(model.to_json())