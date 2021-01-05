def fitness_calc(k):
    import numpy as np
    from chalmers_NN import NeuralNetwork
    from sklearn.datasets import load_linnerud
    data = load_linnerud()
    X = data.data
    y = data.target
    
    X = X/np.max(X,axis=0)
    y = y/np.max(y,axis=0)
    y0 = y[:,0].reshape(len(y),1)
    y1 = y[:,1].reshape(len(y),1)
    
    NN = NeuralNetwork(len(X[0]),1)
    epochs=10
    for i in range(epochs): #trains the NN 1000 times
        NN.train(X, y0, k)
        
    for i in range(5): #trains the NN 100 times
        NN.train(X, y1, k)
    o=NN.feedForward(X)
    fitness_t_1= (1-np.sum(np.abs(y1-o))/len(y1))*100
    from numpy import loadtxt
    X = loadtxt('E:\SUSSEX\ALife\Project\Dataset\data_X_water.csv',delimiter=',')
    y = loadtxt('E:\SUSSEX\ALife\Project\Dataset\data_y_water.csv',delimiter=',')
    
    X = X[:12,:]/np.max(X,axis=0)
    y0 = y[:12,0].reshape(12,1)
    y1 = y[:12,2].reshape(12,1)
    
    NN = NeuralNetwork(len(X[0]),1)
    epochs=300
    for i in range(epochs): #trains the NN 1000 times
        NN.train(X, y0, k)
            
    for i in range(100): #trains the NN 100 times
        NN.train(X, y1, k)
    o=NN.feedForward(X)
    fitness_t_2= (1-np.sum(np.abs(y1-o))/len(y1))*100
    fit=[fitness_t_2,fitness_t_1]
    return np.sum(fit)/len(fit)