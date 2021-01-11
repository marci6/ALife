import numpy as np
import matplotlib.pyplot as plt
#3-layers perceptron
class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 2
        
        #weights
        self.W1 = np.random.uniform(-1,1,size=(self.inputSize, self.hiddenSize)) # weight matrix from input to hidden layer
        self.W2 = np.random.uniform(-1,1,size=(self.hiddenSize, self.outputSize)) #  weight matrix from hidden to output layer
        self.B1 = np.zeros((1, self.hiddenSize)) #bias second layer
        self.B2 = np.zeros((1, self.outputSize)) #bias third layer
        
    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) + self.B1 # dot product of X (input) and first set of weights
        self.z2 = self.activation(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) + self.B2 # dot product of hidden layer (z2) and second set of weights 
        output = self.activation(self.z3)
        return output
    
    def activation(self, s, deriv=False):
        if (deriv == True):
            return 1-np.tanh(s)**2
        return np.tanh(s)
    
    def backward(self, X, y, output,lr):
        # backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.activation(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.activation(self.z2, deriv=True) #applying derivative of activation to z2 error
        
        self.W1 += lr*X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += lr*self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
        self.B2 += lr*np.mean(self.output_delta) # adjusting first set (input -> hidden) biases
        self.B1 += lr*np.mean(self.z2_delta,axis=0) # adjusting second set (hidden -> output) biases
        
    def train(self, X, y,lr):
        output = self.feedForward(X)
        self.backward(X, y, output,lr)
        
# XOR - Data
X = np.array(([0, 0], [1, 0], [0, 1], [1, 1]), dtype=int)
y = np.array(([0], [1], [1], [0]), dtype=int)

Epochs=[]
for test in range(100):
    NN = NeuralNetwork()
    epochs=0
    error=1
    loss=[]
    while error>0.04 and epochs<10000: 
        NN.train(X, y,0.1)
        loss.append(np.mean(np.square(y - NN.feedForward(X))))
        error= np.sum((y-NN.feedForward(X))**2)
        epochs+=1
        
#    plt.plot(loss)  
    if epochs!=10000:
        Epochs.append(epochs)
# Statistics    
avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)