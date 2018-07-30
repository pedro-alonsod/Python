#!/usr/bin/env python

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
import pandas as pd
from keras.preprocessing.text import Tokenizer

import tp4_utils

# parameters of our model
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

#Load training data
train = pd.read_csv("data/allocine_train_clean.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("data/allocine_test_clean.tsv", header=0, delimiter="\t", quoting=3)

# Keras's tokenizer turns each word into a word index, that will be
# used to look up the word's embedding. We first construct its
# vocabulary by fitting it on the training data; max_features defines
# how many words will be taken into account
tokenizer = Tokenizer(num_words=max_features, lower=True, split=" ")
tokenizer.fit_on_texts(train["review"])

# construct mappings from words to indices and vice versa
w2i = tokenizer.word_index
i2w = {i:w for w,i in w2i.items()}

# now we turn each word into its word index; the sequence of indices
# (x_train) will be the input to our network, while y_train is the
# output we train the network on
x_train = tokenizer.texts_to_sequences(train["review"])
x_train = np.asarray(x_train)
y_train = train["sentiment"]

# same for the test data
x_test = tokenizer.texts_to_sequences(test["review"])
x_test = np.asarray(x_test)
y_test = test["sentiment"]

# Because we use batches, each sequence needs to be the same length;
# we therefore "pad" the sequences, which means we create fixed-length
# vector and fill up the positions that are not used with zeros
print 'Pad sequences'
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print 'x_train shape:', x_train.shape
print 'x_test shape:', x_test.shape

print 'Build model...'

# First we build our model. We define a sequential model to which we
# can add network layers
model = Sequential()

# We start off with an embedding layer which maps
# our indices into embeddings of size embedding_dims
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

# We add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a
# sigmoid. This is similar to what we did in the former exercise: as
# we only have two classes, we can just predict a value between 0
# (negative) and positive (1)
model.add(Dense(1, activation='sigmoid'))

# Once our model is constructed, we can compile it
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Now we train our model on the training set and validate on the test
# set
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(x_test, y_test))
