#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import sklearn

from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

MAX_FEATURES = 5000

#Load train and test set
train = pd.read_csv("data/allocine_train_clean.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("data/allocine_test_clean.tsv", header=0,
                   delimiter="\t", quoting=3)

print "Creating features from bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(
    analyzer = "word",
    max_features = MAX_FEATURES
) 

# fit_transform() performs two operations; first, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(train["review"])

# output from vectorizer is a sparse array; our classifier needs a
# dense array
x_train = train_data_features.toarray()

# construct a matrix of two columns (one for positive class, one for
# negative class) where the correct class is indicated with 1 and the
# incorrect one with 0
y_train = np_utils.to_categorical(np.asarray(train["sentiment"]))

# same process for test set
test_data_features = vectorizer.transform(test["review"])
x_test = test_data_features.toarray()
y_test = np_utils.to_categorical(np.asarray(test["sentiment"]))

print "Build model.."

# construct a neural network model with
# 1) an input layer of MAX_FEATURES neurons
# 2) a hidden layer with 4 neurons, with relu activation applied
# 3) an output layer of 2 neurons, with softmax applied to ensure
#    a probability distribution
model = Sequential()
model.add(Dense(4, input_shape=(MAX_FEATURES,)))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

# the model uses cross-entropy as a loss function, finds the best
# parameters using stochastic gradient descent, and prints accuracy
# metrics
model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['acc'])

# now we train the model by feeding it a batch of 8 training samples
# at a time. When we get to the end of the training set, we repeat the
# process, and we do this 10 times (epochs). We validate our model on
# the test set (remember though: when we test different model
# parameters we should make use of a validation set separate from the
# test set)
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=10, batch_size=8)
