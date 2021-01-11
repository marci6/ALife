class NeuralNetwork(object):
    def __init__(self,W1,W2,W3,W4,B1,B2,B3,B4):
        import numpy as np
        # weights
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W4 = W4
        # bias
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.B4 = B4
        
        self.v1=np.zeros((len(W1), len(W1[0])))
        self.v2=np.zeros((len(W2), len(W2[0])))
        self.v3=np.zeros((len(W3), len(W3[0])))
        self.v4=np.zeros((len(W4), len(W4[0])))
        
    def feedForward(self, X):
        import numpy as np
        #forward propogation through the network
        self.z = self.sigmoid(np.dot(X, self.W1)  + self.B1) 
        self.z2 = self.sigmoid(np.dot(self.z, self.W2) + self.B2)
        self.z3 = self.sigmoid(np.dot(self.z2, self.W3) + self.B3)
        output = self.sigmoid(np.dot(self.z3, self.W4) + self.B4)
        return output
    
    def sigmoid(self, x, deriv=False):
        import numpy as np
        if (deriv == True):
            return 1-np.tanh(x)**2
        return np.tanh(x)
    
    def backward(self, X, y, output, lr):
        import numpy as np
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z3_error = self.output_delta.dot(self.W4.T) 
        self.z3_delta = self.z3_error * self.sigmoid(self.z3, deriv=True) 
        
        self.z2_error = self.z3_delta.dot(self.W3.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)
        
        self.z_error = self.z2_delta.dot(self.W2.T)
        self.z_delta = self.z_error * self.sigmoid(self.z, deriv=True)
        
        m=0.9
        
        self.v1= self.v1*m + (1-m)*X.T.dot(self.z_delta)
        self.v2=self.v2*m + (1-m)*self.z.T.dot(self.z2_delta)
        self.v3=self.v3*m + (1-m)*self.z2.T.dot(self.z3_delta)
        self.v4=self.v4*m + (1-m)*self.z3.T.dot(self.output_delta)
        
        self.W1 += lr*self.v1
        self.W2 += lr*self.v2
        self.W3 += lr*self.v3
        self.W4 += lr*self.v4
        
        self.B4 += lr*np.mean(self.output_delta)
        self.B3 += lr*np.mean(self.z3_delta,axis=0)
        self.B2 += lr*np.mean(self.z2_delta,axis=0)
        self.B1 += lr*np.mean(self.z_delta,axis=0)
        
    def train(self, X, y, lr):
        output = self.feedForward(X)
        self.backward(X, y, output, lr)