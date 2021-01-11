class NeuralNetwork(object):
    def __init__(self,W1,W2,B1,B2):      
        #weights
        self.W1 = W1 
        self.W2 = W2
        self.B1 = B1
        self.B2 = B2
    def feedForward(self, X):
        import numpy as np
        #forward propogation through the network
        self.z = np.dot(X, self.W1) +self.B1#dot product of X (input) and first set of weights 
        self.z2 = self.activation(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) + self.B2 #dot product of hidden layer (z2) and second set of weights 
        output = self.activation(self.z3)
        return output
    
    def activation(self, x, deriv=False):
        import numpy as np
        if (deriv == True):
            return 1-np.tanh(x)**2
        return np.tanh(x)
    
    def backward(self, X, y, output, lr):
        #backward propogate through the network
        import numpy as np
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.activation(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.activation(self.z2, deriv=True) #applying derivative of activation to z2 error
        
        self.W1 += lr*X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += lr*self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
        self.B2 += lr*np.mean(self.output_delta)
        self.B1 += lr*np.mean(self.z2_delta,axis=0)
        
    def train(self, X, y,lr):
        output = self.feedForward(X)
        self.backward(X, y, output, lr)