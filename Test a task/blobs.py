# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:42:54 2020

@author: MARCELLOCHIESA
"""
import numpy as np
from chalmers_NN import NeuralNetwork
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=3, random_state=10)

X = X/np.max(X, axis=0)
y = y.reshape(200,1)

X_train = X[:150,:]
y_train=y[:150,:]
X_test=X[150:,:]
y_test=y[150:,:]

NN = NeuralNetwork(len(X[0]),1)
k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
lr=0.1    
epochs=10
for i in range(epochs): 
    NN.train(X_train, y_train, lr*k)
    print('Train accuracy:',(1-np.sum(np.abs(y_train-NN.feedForward(X_train)))/len(y_train))*100)
print('Test accuracy:',(1-np.sum(np.abs(y_test-NN.feedForward(X_test)))/len(y_test))*100)

from matplotlib import pyplot


# create a scatter plot of points colored by class value
def plot_samples(X, y, classes=3):
	# plot points for each class
	for i in range(classes):
		# select indices of points with each class label
		samples_ix = np.where(y == i)
		# plot points for this class with a given color
		pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])


# scatter plot of samples
plot_samples(X_train, y_train)
pyplot.figure()
plot_samples(X_test, y_test)
# plot figure
pyplot.show()