# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:38:54 2020

@author: MARCELLOCHIESA
"""
import numpy as np
from chalmers_NN import NeuralNetwork
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[0],[0],[1]])
k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
lr=1    
NN = NeuralNetwork(2,1)
for e in range(10):
    NN.train(X,y,lr*k)
print('Task accuracy:',(1-np.sum(np.abs(y-NN.feedForward(X)))/len(y))*100)