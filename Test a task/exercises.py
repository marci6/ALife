# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:45:31 2020

@author: MARCELLOCHIESA
"""

from sklearn.datasets import load_linnerud
import numpy as np
from chalmers_NN import NeuralNetwork
data = load_linnerud()
X = data.data
y = data.target

X = X/np.max(X,axis=0)
y = y/np.max(y,axis=0)

X_train=X[:15,:]
y_train=y[:15,:]

X_test=X[15:,:]
y_test=y[15:,:]

NN = NeuralNetwork(len(X[0]),1)
k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
lr=0.1 
epochs=200
for i in range(len(y[0])):
    y0 = y_train[:,i].reshape(len(y_train),1)
    NN = NeuralNetwork(len(X[0]),1)
    for i in range(epochs): #trains the NN 1000 times
        NN.train(X_train, y0, lr*k)
    print('Task accuracy:',(1-np.sum(np.abs(y0-NN.feedForward(X_train)))/len(y0))*100)


print('Test accuracy:',(1-np.sum(np.abs(y_test[:,2]-NN.feedForward(X_test)))/5)*100)