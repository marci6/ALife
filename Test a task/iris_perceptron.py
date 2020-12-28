# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:24:58 2020

@author: MARCELLOCHIESA
"""

import numpy as np
import matplotlib.pyplot as plt
from perceptron_NN import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()

X = data.data
y = data.target/2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
y_test = y_test.reshape(len(y_test),1)
y_train = y_train.reshape(len(y_train),1)
NN = NeuralNetwork(4,1)
loss=[]
for e in range(400):
    NN.train(X_train,y_train)
    loss.append(np.mean(np.abs(y_train-(NN.feedForward(X_train)))))
print('Mean accuracy:',(1-np.sum(np.abs(y_test-NN.feedForward(X_test)))/len(y_test))*100,'%')
plt.plot(loss)
plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], color='red')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue')
