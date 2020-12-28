# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:25:23 2020

@author: MARCELLOCHIESA
"""

import numpy as np
from chalmers_NN import NeuralNetwork

X=np.array([[1,1,1,1,1],[0,0,0,0,0],[0,1,1,1,0],[1,1,0,0,0],[1,0,1,0,1],\
            [0,1,1,0,0],[0,1,1,1,1],[0,1,0,0,0],[1,1,0,0,1],[1,0,0,1,0],[1,0,1,1,0],[0,0,0,1,0]])
task_set=np.array([[1,0,0,0,1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0],[0,1,1,1,0,1,1,1,0,1,0,0],\
             [0,0,0,1,0,0,0,1,1,1,0,0],[1,0,1,0,0,1,1,1,0,1,0,1],[0,1,0,1,0,1,0,1,1,0,0,0],\
             [1,0,1,1,1,0,1,0,1,1,1,0],[0,1,0,1,1,1,0,1,1,1,0,1]])

NN = NeuralNetwork(5,1)
k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
epochs=10
lr=1
for y in task_set:
    for e in range(epochs):
        NN.train(X,y.reshape(12,1),lr*k)
    print('Task accuracy:',(1-np.sum(np.abs(y.reshape(12,1)-NN.feedForward(X)))/12)*100)


