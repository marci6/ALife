class NeuralNetwork(object):
    def __init__(self,i,j,k):
        import numpy as np
        #parameters
        self.inputSize = i
        self.outputSize = k
        self.hiddenSize = j
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (ixj) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (jxk) weight matrix from hidden to output layer
        self.B1 = np.zeros((1,self.hiddenSize))
        self.B2 = np.zeros((1,self.outputSize))
        self.v1=np.zeros((self.inputSize, self.hiddenSize))
        self.v2=np.zeros((self.hiddenSize, self.outputSize))
        
    def feedForward(self, X):
        import numpy as np
        #forward propogation through the network
        self.z = np.dot(X, self.W1) + self.B1 #dot product of X (input) and first set of weights
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) + self.B2 #dot product of hidden layer (z2) and second set of weights 
        output = self.sigmoid(self.z3)
        return output
    
    def sigmoid(self, x, deriv=False):
        import numpy as np
        if (deriv == True):
            return self.sigmoid(x)*(1 - self.sigmoid(x))
        return 1/(1 + np.exp(-x))
    
    def backward(self, X, y, output, lr):
        import numpy as np
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.v1= self.v1*0.9 + 0.1*X.T.dot(self.z2_delta)
        self.v2=self.v2*0.9 + 0.1*self.z2.T.dot(self.output_delta)
        
        self.W1 += lr*self.v1 # adjusting first set (input -> hidden) weights
        self.W2 += lr*self.v2 # adjusting second set (hidden -> output) weights
        
        self.B1 += lr*np.mean(self.z2_delta,axis=0)
        self.B2 += lr*np.mean(self.output_delta,axis=0)
        
    def train(self, X, y, lr):
        output = self.feedForward(X)
        self.backward(X, y, output, lr)
