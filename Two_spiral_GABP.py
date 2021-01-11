import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from NN_5layers import NeuralNetwork

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))
# Data
X, y = twospirals(300)

X = StandardScaler().fit_transform(X)
#PLOT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=21)
#h = .02
#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

## just plot the dataset first
#cm = plt.cm.RdBu
#cm_bright = ListedColormap(['#FF0000', '#0000FF'])

## Plot the training points
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
## Plot the testing points
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')

y_test =y_test.reshape(len(y_test),1)
y_train =y_train.reshape(len(y_train),1)
# Dimension NN are N,M,K
N=2 #input size
M=20 #hidden size
K=1 # output size
print('Start')
SizePop=50 #population size
SizeGen = N*M+2*M*M+M*K+3*M+K #genome size
D=5
tours=[]
Epochs=[]
for test in range(1):    
    # initialize Population
    Pop=np.random.randn(SizePop,SizeGen)
    fitness=np.zeros(SizePop)
    best_fit=[]
    # fitness evaluation
    for p in range(SizePop):
        Gene = Pop[p,:]
        W1 = Gene[:N*M].reshape(N,M)
        B1 = Gene[N*M:N*M+M].reshape(1,M)
        W2 = Gene[N*M+M:N*M+M*M+M].reshape(M,M)
        B2 = Gene[N*M+M*M+M:N*M+M*M+2*M].reshape(1,M)
        W3 = Gene[N*M+M*M+2*M:N*M+2*M*M+2*M].reshape(M,M)
        B3 = Gene[N*M+2*M*M+2*M:N*M+2*M*M+3*M].reshape(1,M)
        W4 = Gene[N*M+2*M*M+3*M:N*M+2*M*M+3*M+M*K].reshape(M,K)
        B4 = Gene[N*M+2*M*M+3*M+M*K:].reshape(1,K)
        NN = NeuralNetwork(W1,W2,W3,W4,B1,B2,B3,B4)
        output=NN.feedForward(X_train)
        fitness[p] = 1/(np.sum(np.abs(y_train-np.round(output))))    
        
    NoTournaments=0
    errorGA=1000
    while errorGA>150 and NoTournaments!=1000:
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
            Pop[Loser,k]=np.min([10,Pop[Loser,k]]) 
            Pop[Loser,k]=np.max([-10,Pop[Loser,k]]) 
        # measure new fitness
        for p in range(SizePop):
            Gene = Pop[p,:]
            W1 = Gene[:N*M].reshape(N,M)
            B1 = Gene[N*M:N*M+M].reshape(1,M)
            W2 = Gene[N*M+M:N*M+M*M+M].reshape(M,M)
            B2 = Gene[N*M+M*M+M:N*M+M*M+2*M].reshape(1,M)
            W3 = Gene[N*M+M*M+2*M:N*M+2*M*M+2*M].reshape(M,M)
            B3 = Gene[N*M+2*M*M+2*M:N*M+2*M*M+3*M].reshape(1,M)
            W4 = Gene[N*M+2*M*M+3*M:N*M+2*M*M+3*M+M*K].reshape(M,K)
            B4 = Gene[N*M+2*M*M+3*M+M*K:].reshape(1,K)
            NN = NeuralNetwork(W1,W2,W3,W4,B1,B2,B3,B4)
            output=NN.feedForward(X_train)
            fitness[p] = 1/(np.sum(np.abs(y_train-np.round(output)))) 
        errorGA=1/np.max(fitness)
        NoTournaments+=1
    
    
    Gene = Pop[np.argmax(fitness),:]
    W1 = Gene[:N*M].reshape(N,M)
    B1 = Gene[N*M:N*M+M].reshape(1,M)
    W2 = Gene[N*M+M:N*M+M*M+M].reshape(M,M)
    B2 = Gene[N*M+M*M+M:N*M+M*M+2*M].reshape(1,M)
    W3 = Gene[N*M+M*M+2*M:N*M+2*M*M+2*M].reshape(M,M)
    B3 = Gene[N*M+2*M*M+2*M:N*M+2*M*M+3*M].reshape(1,M)
    W4 = Gene[N*M+2*M*M+3*M:N*M+2*M*M+3*M+M*K].reshape(M,K)
    B4 = Gene[N*M+2*M*M+3*M+M*K:].reshape(1,K)
    NN = NeuralNetwork(W1,W2,W3,W4,B1,B2,B3,B4)
    
#    print('Pre BP error', np.mean(np.abs(y_test - NN.feedForward(X_test))))
#    print('Accuracy pre : ',(1-np.sum(np.abs(y_test-NN.feedForward(X_test)))/len(y_test))*100,'%')
    
    epochs=0
    error=1000
    loss=[]
    #fine tuning
    while error>12 and epochs!=100000:
        NN.train(X_train,y_train,0.0001)
        error=np.sum(np.abs(y_test-np.round(NN.feedForward(X_test))))
        loss.append(error)
        epochs+=1
    if epochs!=100000:
        Epochs.append(epochs)
        tours.append(NoTournaments)

plt.figure()
plt.title('NMN Encoder')
plt.plot(best_fit)
plt.xlabel('Tournaments')
plt.ylabel('Fitness')
plt.figure()
plt.title('NMN Encoder')
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('TSS Error')

#print('Post BP error', np.sum((y - NN.feedForward(X))**2))
#print('Post BP error', np.mean(np.abs(y_test - NN.feedForward(X_test))))
#print('Accuracy post : ',(1-np.sum(np.abs(y_test-NN.feedForward(X_test)))/len(y_test))*100,'%')
# statistics       
avg_tour=np.mean(tours)
std_tour=np.std(tours)

avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of tournaments %.1f'%avg_tour)
print('Standard deviation of tournaments %.1f'%std_tour)
print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)