import numpy as np
import matplotlib.pyplot as plt
from NN_moment_GA import NeuralNetwork


def MNMencoder(M):
 X = np.zeros((M,M)) 
 for i in range(M):        
     X[i,i]=1
 return X

# MNM Encoder
S=16 # Input size
X = MNMencoder(S)
np.random.shuffle(X)
y = X
# Dimension NN are N,M,K
N=S
M=4
K=S

print('Start')
SizePop=20 # Population size
SizeGen = N*M+M*K+K+M # genome size
D=5 # Deme size

tours=[]
Epochs=[]
for test in range(100):
    # initialize Population
    Pop=np.random.randn(SizePop,SizeGen)*10 # initialization
    fitness=np.zeros(SizePop)
    best_fit=[]
    # fitness evalutation
    for p in range(SizePop):
        Gene = Pop[p,:]
        W1 = Gene[:N*M].reshape(N,M)
        B1 = Gene[N*M:N*M+M].reshape(1,M)
        W2 = Gene[N*M+M:N*M+M*K+M].reshape(M,K)
        B2 = Gene[N*M+M*K+M:].reshape(1,K)
        NN = NeuralNetwork(W1,W2,B1,B2)
        output=NN.feedForward(X)
        fitness[p] = 1/(np.sum((y-output)**2))    
        
    NoTournaments=0
    errorGA=100
    while errorGA>25 and NoTournaments!=1000:
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
          if np.random.rand()<0.8:
            Pop[Loser,k]=Pop[Winner,k]
          #mutation
          if np.random.rand()<0.05:
            Pop[Loser,k]+=np.random.randn()
            Pop[Loser,k]=np.min([100,Pop[Loser,k]]) 
            Pop[Loser,k]=np.max([-100,Pop[Loser,k]]) 
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
        errorGA=1/np.max(fitness)
        NoTournaments+=1
    
    Gene = Pop[np.argmax(fitness),:]
    W1 = Gene[:N*M].reshape(N,M)
    B1 = Gene[N*M:N*M+M].reshape(1,M)
    W2 = Gene[N*M+M:N*M+M*K+M].reshape(M,K)
    B2 = Gene[N*M+M*K+M:].reshape(1,K)
    NN = NeuralNetwork(W1,W2,B1,B2)
    
#    print('\nPre BP error',np.sum((y - NN.feedForward(X))**2))

    epochs=0
    error=1
    loss=[]
    # fine tuning
    while error>0.04 and epochs!=10000:
        NN.train(X,y,0.5)
        error=np.sum((y-NN.feedForward(X))**2)
        loss.append(error)
        epochs+=1
    if epochs!=10000:
        Epochs.append(epochs)
        tours.append(NoTournaments)
    
#    plt.figure()
#    plt.title('NMN Encoder')
#    plt.plot(best_fit)
#    plt.xlabel('Tournaments')
#    plt.ylabel('Fitness')
#    plt.figure()
#    plt.title('NMN Encoder')
#    plt.plot(loss)
#    plt.xlabel('Epochs')
#    plt.ylabel('TSS Error')
#    print('Post BP error', np.sum((y - NN.feedForward(X))**2))
    
# statistics
avg_tour=np.mean(tours)
std_tour=np.std(tours)

avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of tournaments %.1f'%avg_tour)
print('Standard deviation of tournaments %.1f'%std_tour)
print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)