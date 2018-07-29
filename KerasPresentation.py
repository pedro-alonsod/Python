from keras.models import Sequential
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
# from keras.models import Functional
model = Sequential()



from keras.layers.core import Dense, Activation


#generated train data
xTain = np.random.random((100, 100, 100, 3))
yTrain = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

xTests = np.random.random((20, 100, 100, 3))
yTests = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

print(xTain, "xTain")
print(yTrain, "yTrain")
print(xTests, "xTests")
print(yTests, "yTests")

model.add(Conv2D(32, (3,3),activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# model.add(Activation("softmax"))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(xTain, yTrain, batch_size=32, epochs=10)
score = model.evaluate(xTests, yTests, batch_size=32)

print(model.summary())
print(score, "Score")

from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

print(model.summary())

model.fit(xTain, yTrain, nb_epoch=5, batch_size=32)

#model.train_on_batch(X_batch, Y_batch)

objective_score = model.evaluate(xTests, yTests, batch_size=32)

classes = model.predict_classes(xTests, batch_size=32)
proba = model.predict_proba(xTests, batch_size=32)

# Non existant anymore graph model with one input and two outputs
# graph = Graph()
# graph.add_input(name='input', input_shape=(32,))
# graph.add_node(Dense(16), name='dense1', input='input')
# graph.add_node(Dense(4), name='dense2', input='input')
# graph.add_node(Dense(4), name='dense3', input='dense1')
# graph.add_output(name='output1', input='dense2')
# graph.add_output(name='output2', input='dense3')
# graph.compile(optimizer='rmsprop', loss={'output1':'mse',
# 'output2':'mse'})
# history = graph.fit({'input':X_train, 'output1':y_train,
# 'output2':y2_train}, nb_epoch=10)

model2 = Sequential()
model2.add(Dense(units=64, input_dim=64))
model2.add(Activation('relu'))
model2.add(Dense(units=10))
model2.add(Activation('relu'))

model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print(model2.summary())