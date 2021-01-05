
import numpy as np

# XOR(AND(OR))
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],[0,1,1],[1,0,0]), dtype=int)
y = np.array(([0], [1], [1],[0],[0]), dtype=int)


class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize) # (3x3) weight matrix from hidden to output layer
        self.W3 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (3x2)
        self.z2 = self.activation(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)
        self.z4 = self.activation(self.z3)
        self.z5= np.dot(self.z4,self.W3)
        output = self.activation(self.z5)
        return output
    
    def activation(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def backward(self, X, y, output):
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.activation(output, deriv=True)
        
        self.z4_error = self.output_delta.dot(self.W3.T) #z4 error: how much our hidden layer weights contribute to output error
        self.z4_delta = self.z4_error * self.activation(self.z4, deriv=True)
        
        self.z2_error = self.z4_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.activation(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.z4_delta) # adjusting second set (hidden -> output) weights
        self.W3 += self.z4.T.dot(self.output_delta)
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()
epoch=1000
for i in range(epoch): #trains the NN 1000 times
    if (i % 100 == 0):
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)
    
Test=np.array(([1,0,1],[1,1,0],[1,1,1]),dtype=int)

print("\nInput: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: \n" + str(np.round(NN.feedForward(Test)).astype(int)))  
