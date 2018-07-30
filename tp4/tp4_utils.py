#!/usr/bin/env python

import numpy as np

def calcSim(word, w2i, i2w, matrix, nbest=20):
    #normalize matrix
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i] / np.linalg.norm(matrix[i])
    
    i = w2i[word]
    cosineVector = np.dot(matrix[i],matrix.T)
    cosList = [(cosineVector[i],i) for i in range(cosineVector.shape[0])]
    cosList.sort()
    cosList.reverse()
    mostSimList = [(i2w[j],i) for i,j in cosList[0:nbest]]
    for i,j in mostSimList:
        print i,j

        
def sample(preds, temperature=0.7):
    # helper function to sample an index from a probability array                                                                                                                                                                                                                                                           
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
