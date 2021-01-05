# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:31:21 2020

@author: MARCELLOCHIESA
"""

import numpy as np
from chalmers_NN import NeuralNetwork
def load_csv(filename):
    from csv import reader
    dataset = list()
    with open('E:\SUSSEX\ALife\Project\ALife\sonar_data.csv', 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

filename = 'sonar_data.csv'
dataset = load_csv(filename)
dataset=dataset[1:len(dataset)-1]
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
dataset=np.asarray(dataset)

y = dataset[:,-1]
y = y.reshape(len(y),1)
X = dataset[:,0:len(dataset[0])-1]

X_train=X[:150,:]
X_test=X[150:,:]

y_train=y[:150]
y_test=y[150:]

NN = NeuralNetwork(len(X[0]),1)
k=np.array([4, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0])
lr=1
for e in range(150 00):
    NN.train(X_train,y_train,lr*k)
print('Test accuracy:',(1-np.sum(np.abs(y_test-NN.feedForward(X_test)))/len(y_test))*100)
    
    