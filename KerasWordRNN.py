import numpy as np
import data.load

train_set, valid_set, dicts = data.load.atisfull()
w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']

train_x, _, train_label = train_set
val_x, _, val_label = valid_set

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

# For conlleval script
words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
labels_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
labels_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]

n_classes = len(idx2la)
n_vocab = len(idx2w)