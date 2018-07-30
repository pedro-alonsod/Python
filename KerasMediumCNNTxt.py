import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
import os
import sklearn.metrics
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline  #ipython magic

output_dir = './model_output/imdb_deep_net'
epochs = 60
batch_size = 128
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'
n_dense = 64
dropout = 0.5

import sys
if sys.version_info[0] < 3:
	print(sys.version_info[0])
	(X_train, y_train), (X_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)
else:
	print(sys.version_info[0])
	(X_train, y_train), (X_valid, y_valid) = imdb.load_data(nb_words=n_unique_words, skip_top=n_words_to_skip) 
	import theano.ifelse
	nb_epoch = 60   

#(X_train, y_train), (X_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)

print(X_train, "X_train")

for x in X_train[0:6]:
    print(len(x), "len(X_train)", x)

word_index = keras.datasets.imdb.get_word_index()
# v+3 so we push the words 3 positions.
word_index = {k : (v+3) for k,v in word_index.items()}
# Now we fill in some keywords for the first 3 indexes as seen below.
word_index['PAD'] = 0
word_index['START'] = 1
word_index['UNK'] = 2

index_word = {v: k for k, v in word_index.items()}

review = ' '.join(index_word[id] for id in X_train[0])
print(review)

(all_X_train, _), (all_X_valid, _) = imdb.load_data()
full_review = ' '.join(index_word[id] for id in all_X_train[0])
print(full_review)


X_train = pad_sequences(X_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
X_valid = pad_sequences(X_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)


for x in X_train[0:6]:
    print(len(x))

review = ' '.join(index_word[id] for id in X_train[0])
print(review)

review = ' '.join(index_word[id] for id in X_train[5])
print(review)


model = Sequential()
model.add(Embedding(n_unique_words, n_dim, 
                    input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()

modelcheckpoint = ModelCheckpoint(filepath=output_dir+'/weights.{epoch:02d}.hdf5')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


if sys.version_info[0] < 3:
	model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[modelcheckpoint])
else:
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.2, callbacks=[modelcheckpoint])	

y_hat = model.predict_proba(X_valid)

pct_auc = roc_auc_score(y_valid, y_hat) * 100
print('{:0.2f}'.format(pct_auc))

fpr, tpr, _ = sklearn.metrics.roc_curve(y_valid, y_hat)
roc_auc = sklearn.metrics.auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Create a Pandas DataFrame to hold y_hat and y values.
ydf = pd.DataFrame(list(zip(y_hat[:,0], y_valid)),
                   columns=['y_hat', 'y'])
# Print the first 10 rows.
print(ydf.head(10))


# Read a full review. In that case, the 1st one.# Read a full review. In that case, the 1st 
index = 0
review = ' '.join(index_word[id] for id in all_X_valid[index])
print(review)