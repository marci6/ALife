# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:48:47 2020

@author: MARCELLOCHIESA
"""
import numpy as np
from chalmers_NN import NeuralNetwork
from numpy import loadtxt

X = loadtxt('E:\SUSSEX\ALife\Project\Dataset\data_X_water.csv',delimiter=',')
y = loadtxt('E:\SUSSEX\ALife\Project\Dataset\data_y_water.csv',delimiter=',')

X = X/np.max(X,axis=0)
Xt = X[250:349,:]
X = X[:250,:]


k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
lr=0.1 
epochs=300
for i in range(len(y[0])):
    NN = NeuralNetwork(len(X[0]),1)
    y0 = y[:250,i].reshape(250,1)
    for i in range(epochs): #trains the NN 1000 times
        NN.train(X, y0, lr*k)
    print('Task accuracy:',(1-np.sum(np.abs(y0-NN.feedForward(X)))/len(y0))*100)
    #%%
NN = NeuralNetwork(len(X[0]),1)
y0 = y[:250,12].reshape(250,1)
for i in range(1000): #trains the NN 1000 times
    NN.train(X, y0, lr*k)
y0 = y[250:,12].reshape(349-250,1)
print('Test accuracy:',(1-np.sum(np.abs(y0-NN.feedForward(Xt)))/len(y0))*100)
    
