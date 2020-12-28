# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:31:44 2020

@author: marci
"""

import numpy as np

# XOR
X = np.array(([0, 0], [1, 0], [0, 1], [1, 1]), dtype=int)
y = np.array(([0], [1], [1], [0]), dtype=int)


class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 2
        
        #weights
        self.W1 = np.random.uniform(-1,1,size=(self.inputSize, self.hiddenSize)) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.uniform(-1,1,size=(self.hiddenSize, self.outputSize)) # (3x1) weight matrix from hidden to output layer
        
    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (3x2)
        self.z2 = self.activation(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.activation(self.z3)
        return output
    
    def activation(self, s, deriv=False):
        if (deriv == True):
            return 1-np.tanh(s)**2
        return np.tanh(s)
    
    def backward(self, X, y, output):
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.activation(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.activation(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()
epochs=1000
for i in range(epochs): #trains the NN 1000 times
    if (i % 100 == 0):
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)
    
print("\nInput: " + str(X))
print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: " + str(np.round(NN.feedForward(X)).astype(int)))  