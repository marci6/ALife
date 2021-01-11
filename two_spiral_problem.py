import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from layer_NN import NeuralNetwork

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))
#Data
X, y = twospirals(150)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=21)

# PLOT SPIRALS
#h = .02
#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#
## just plot the dataset first
#cm = plt.cm.RdBu
#cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#plt.figure()
## Plot the training points
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
## Plot the testing points
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')

y_test =y_test.reshape(len(y_test),1)
y_train =y_train.reshape(len(y_train),1)
tours=[]
Epochs=[]

for test in range(20):
    NN = NeuralNetwork(2,20,20,20,1)
    epochs=0
    error=1000
    loss=[]
    while error>12 and epochs!=100000:
        NN.train(X_train,y_train,0.0001)
        error=np.sum(np.abs(y_test-np.round(NN.feedForward(X_test))))
        loss.append(error)
        epochs+=1
    if epochs!=100000:
        Epochs.append(epochs)

#plt.figure()
#plt.plot(loss)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')    
#print('Accuracy test: ',(1-np.sum(np.abs(y_test.reshape(len(y_test),1)-np.round(NN.feedForward(X_test))))/len(y_test))*100,'%')
#print('Accuracy train: ',(1-np.sum(np.abs(y_train.reshape(len(y_train),1)-np.round(NN.feedForward(X_train))))/len(y_train))*100,'%')
#print('Error ', np.mean(np.abs(y_train.reshape(len(y_train),1)-np.round(NN.feedForward(X_train)))))
        
#statistics
avg_epochs=np.mean(Epochs)
std_epochs=np.std(Epochs)

print('Average number of epochs %.1f'%avg_epochs)
print('Standard deviation of epochs %.1f'%std_epochs)