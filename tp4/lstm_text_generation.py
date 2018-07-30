#!/usr/bin/env python

'''Recurrent network character-based language model to generate names of French
villages
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop, Adam
import numpy as np
import random
import sys
import tp4_utils

# Load the data as one long string
text = open('data/communes_france.txt').read().lower()
print('corpus length:', len(text))

# Our model is character-based, so we want a list of possible
# characters
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Cut the text in semi-redundant sequences of maxlen characters. Given
# a string of 40 characters, we want our model to learn to predict the
# next character
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# Our architecture needs vectorized data, so we need to convert our
# characters to numbers. We will create vectors that indicate which
# character is present at a particular position. Each vector contains
# 30 values (the length of our character set), which are all zero
# except for the index of the actual character in that particular
# position. Our sequences are 40 characters long, so we create a
# matrix for of size 40x30. We create such a matrix for all of our
# sequences. This will be our input data (X). Additionally, we create
# a vector that indicates the next character in the sequence. This
# will be the output value we train on (y).
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM. At each timestep, the LSTM predicts
# the next character in the sequence. This is a probability
# distribution over the set of characters
print('Build model...')
model = Sequential()
model.add(LSTM(8, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# We use an advanced optimizer called RMSprop; our output is a
# probability distribution, so cross-entropy is a suitable loss
# function
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# train the model, output generated text after each iteration
for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, epochs=1)

    
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    # We use our trained model to sample 400 characters
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        # This is where the prediction of the next character takes
        # place. The network outputs a probability distribution, from
        # which we sample using a suitable sample function.
        preds = model.predict(x, verbose=0)[0]
        next_index = tp4_utils.sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    generated_list = generated.split('\n')[1:-1]
    print('\n'.join(generated_list))
    print()
