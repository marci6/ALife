# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:15:34 2020

@author: MARCELLOCHIESA
"""

class NeuralNetwork(object):
    def __init__(self,i,j):
        import numpy as np
        #parameters
        self.inputSize = i
        self.outputSize = j
        
        #weights
        self.W1 = np.random.uniform(-1,1,size=(self.inputSize, self.outputSize))
        self.B= np.zeros((1,self.outputSize))
    def feedForward(self, X):
        import numpy as np
        #forward propogation through the network
        self.z = np.dot(X, self.W1) + self.B #dot product of X (input) and first set of weights 
        output = self.activation(self.z) #activation function
        return output
    
    def activation(self, x, deriv=False):
        import numpy as np
        if (deriv == True):
            return self.activation(x)*(1-self.activation(x))
        return 1/(1 + np.exp(-x))
    
    def backward(self, X, y, output):
        import numpy as np
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.activation(output, deriv=True)
        self.W1 += 0.01*X.T.dot(self.output_delta)

        self.B+=0.01*np.mean(y-output)
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)