import numpy as np
from NN_momentum import NeuralNetwork

M=16 # input size
# Data
def MNMencoder(M):
 X = np.zeros((M,M)) 
 for i in range(M):        
     X[i,i]=1
 return X


Epochs=[]
for test in range(10):
    NN = NeuralNetwork(M,4,M)
    X = MNMencoder(M)
    np.random.shuffle(X)
    error=1
    epochs=0
    while error>0.04:
        NN.train(X,X,1)
        error=np.sum((X-(NN.feedForward(X)))**2)
        epochs+=1
        
    Epochs.append(epochs)
    
# Statistics
avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)