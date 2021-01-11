import numpy as np
import matplotlib.pyplot as plt
from NN_GA_step import NeuralNetwork

# XOR - Data
X = np.array(([0, 0], [1, 0], [0, 1], [1, 1]), dtype=int)
y = np.array(([0], [1], [1], [0]), dtype=int)

# Dimension NN are N, M, K
N=2 # Input layer
M=2 # Hidden layer
K=1 # Output layer

print('Start')
SizePop = 20 # Size population
SizeGen = N*M+M*K+K+M # Size genome
D = 5 # Size deme

tours=[]
Epochs=[]
for test in range(100):
    Pop=np.random.randn(SizePop,SizeGen)*5 # initialize population
    fitness=np.zeros(SizePop)
    best_fit=[]
    # fitness evauation
    for p in range(SizePop):
        Gene = Pop[p,:]
        W1 = Gene[:N*M].reshape(N,M)
        B1 = Gene[N*M:N*M+M].reshape(1,M)
        W2 = Gene[N*M+M:N*M+M*K+M].reshape(M,K)
        B2 = Gene[N*M+M*K+M:].reshape(1,K)
        NN = NeuralNetwork(W1,W2,B1,B2)
        output=NN.feedForward(X)
        fitness[p] = 1/(np.sum((y-output)**2))    
    
    NoTournaments = 0
    errorGA = 2
    while  errorGA>1 and NoTournaments<10000:
        best_fit.append(np.max(fitness))
        #selection of two arrangements
        P1=np.random.randint(SizePop)
        P2=(P1+1+np.random.randint(D)) % SizePop
        if fitness[P1]>fitness[P2]:
          Winner=P1
          Loser=P2
        else:
          Winner=P2
          Loser=P1
        for k in range(SizeGen):
          #crossover
          if np.random.rand()<0.5:
            Pop[Loser,k]=Pop[Winner,k]
          #mutation
          if np.random.rand()<0.05:
            Pop[Loser,k]+=np.random.randn()
            Pop[Loser,k]=np.min([10,Pop[Loser,k]]) 
            Pop[Loser,k]=np.max([-10,Pop[Loser,k]]) 
        # measure new fitness
        for p in range(SizePop):
            Gene = Pop[p,:]
            W1 = Gene[:N*M].reshape(N,M)
            B1 = Gene[N*M:N*M+M].reshape(1,M)
            W2 = Gene[N*M+M:N*M+M*K+M].reshape(M,K)
            B2 = Gene[N*M+M*K+M:].reshape(1,K)
            NN = NeuralNetwork(W1,W2,B1,B2)
            output=NN.feedForward(X)
            fitness[p] = 1/(np.sum((y-output)**2))
        errorGA = 1/np.max(fitness)
        NoTournaments+=1
    
    # take best set of weights and biases
    Gene = Pop[np.argmax(fitness),:]
    W1 = Gene[:N*M].reshape(N,M)
    B1 = Gene[N*M:N*M+M].reshape(1,M)
    W2 = Gene[N*M+M:N*M+M*K+M].reshape(M,K)
    B2 = Gene[N*M+M*K+M:].reshape(1,K)
    
    NN = NeuralNetwork(W1,W2,B1,B2)
    
#    print('\nPre BP error',np.sum((y - NN.feedForward(X)))**2)
#    print('Accuracy pre : ',(1-np.sum(np.abs(y-NN.feedForward(X)))/len(y))*100,'%')
    epochs=0
    error=1
    loss=[]
    # fine tuning
    while error>0.04 and epochs<10000:
        NN.train(X,y,0.1)
        error=np.sum((y-NN.feedForward(X))**2)
        loss.append(np.sum(np.square(y-NN.feedForward(X))))
        epochs+=1
    
#    plt.figure()
#    plt.title('XOR Problem GA/BP')    
#    plt.plot(loss)
#    plt.xlabel('Epochs')
#    plt.ylabel('TSS error')   
#    plt.figure()
#    plt.title('XOR Problem GA/BP')
#    plt.plot(best_fit)
#    plt.xlabel('Tournaments')
#    plt.ylabel('Fitness')
#    print('Post BP error', np.sum((y - NN.feedForward(X)))**2)
#    print('Accuracy post : ',(1-np.sum(np.abs(y-NN.feedForward(X)))/len(y))*100,'%')
    if epochs!=10000:
        Epochs.append(epochs)
        tours.append(NoTournaments)
    
avg_tour=np.mean(tours)
std_tour=np.std(tours)

avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of tournaments %.1f'%avg_tour)
print('Standard deviation of tournaments %.1f'%std_tour)
print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)