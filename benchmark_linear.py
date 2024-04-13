#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:01:02 2024

@author: jing
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming you have your time series data in arrays 'X' and 'y'
# X should be a 2D array where each row represents a sample and each column represents a feature
# y should be a 1D array representing the target values (i.e., the values you want to predict)


name = ['BUR', 'SFO', 'SJC', 'LAX', 'OAK', 'ONT', 'SAN']
train_y = np.load('train_y.npy')
test_y = np.load('test_y.npy')
input_X = []
temp = []
for i in range(7):
    temp = []
    for p in range(12):
        temp.append(train_y[0][i][p])
    for t in range(1,len(train_y)):
        temp.append(train_y[t][i][-1])
    for j in range(1,12):
        temp.append(test_y[0][i][j])
    input_X.append(temp)

input_X = np.array(input_X).reshape(-1,1)

# Convert data into sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)
X_Train = []
X_Val = []
X_Test = []
y_Train = []
y_Val = []
y_Test = []
for i in range(7):
    data_scaled = input_X[52*i:52*(i+1)]
    n_steps = 8 
    n_features = 1  
    X, y = create_sequences(data_scaled, n_steps)
    # Reshape data for LSTM input (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # Split data into training and testing sets
    train_size = 32
    X_train, X_test = X[:train_size],X[train_size:]
    y_train, y_test = y[:train_size],y[train_size:]
    
    X_Train.append(X_train)

    X_Test.append(X_test)
    y_Train.append(y_train)

    y_Test.append(y_test)

X_Train = np.array(X_Train).reshape(32*7,8)
X_Test = np.array(X_Test).reshape(12*7,8)

y_Train = np.array(y_Train).reshape(32*7)
y_Test = np.array(y_Test).reshape(12*7)


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_Train, y_Train)

# Make predictions
predictions = model.predict(X_Test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_Test, predictions)
print("Mean Squared Error:", mse)
for i in range(7):
    plt.figure()
    plt.plot(predictions[i*12:(i+1)*12])
    plt.plot(y_Test[i*12:(i+1)*12])
    plt.xlabel(name[i])
test_y_predicted = np.array(predictions).reshape(1,7,12)
np.save('test_y_predicted_linear.npy',test_y_predicted)
test_y