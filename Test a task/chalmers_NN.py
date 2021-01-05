class NeuralNetwork(object):
    def __init__(self,i,j):
        import numpy as np
        #parameters
        self.inputSize = i
        self.outputSize = j
        
        self.W1 = np.random.uniform(-1,1,size=(self.inputSize, self.outputSize)) 
        self.W1[-1,:]=0
        
    def feedForward(self, X):
        import numpy as np
        #forward propogation through the network
        self.z = np.dot(X, self.W1)  #dot product of X (input) and first set of weights (3x2)
        output = self.activation(self.z) #activation function
        return output
    
    def activation(self, x, deriv=False):
        import numpy as np
        if (deriv == True):
            return self.activation(x)*(1-self.activation(x))
        return 1/(1 + np.exp(-x))
    
    def delta_func(self,k,a,o,t,w):
        import numpy as np
        r=len(w)
        c=len(w[0])
        delta_w=np.zeros((r,c))
        for e in range(len(t)):
            for i in range(r):
                for j in range(c):
                    delta_w[i,j]+=k[0]*(k[1]*w[i,j]+k[2]*a[e,i]+k[3]*o[e,j]+k[4]*t[e,j]+k[5]*w[i,j]*a[e,i]+k[6]*w[i,j]*o[e,j]+k[7]*w[i,j]*t[e,j]+k[8]*a[e,i]*o[e,j]+k[9]*a[e,i]*t[e,j]+k[10]*o[e,j]*t[e,j])
        return delta_w 
    
    def backward(self, X, y, output,k):
        import numpy as np
        self.W1 += self.delta_func(k,X,output,y,self.W1) # adjusting first set (input -> hidden) weights
        self.W1 = np.minimum(20*np.ones((self.inputSize,self.outputSize)),self.W1)
        self.W1 = np.maximum(-20*np.ones((self.inputSize,self.outputSize)),self.W1)

        
    def train(self, X, y,k):
        output = self.feedForward(X)
        self.backward(X, y, output,k)