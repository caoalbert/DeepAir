#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:27:13 2024

@author: jing
"""



import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam



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

output_y = []


scaler = MinMaxScaler()
data_scaled_7 = scaler.fit_transform(np.array(input_X).reshape(-1,1))

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
    data_scaled = data_scaled_7[52*i:52*(i+1)]
    n_steps = 8 
    n_features = 1  
    X, y = create_sequences(data_scaled, n_steps)
    # Reshape data for LSTM input (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # Split data into training and testing sets
    train_size = int(32*0.8)
    val_size = 32
    X_train, X_val, X_test = X[:train_size],X[train_size:val_size], X[val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
    
    X_Train.append(X_train)
    X_Val.append(X_val)
    X_Test.append(X_test)
    y_Train.append(y_train)
    y_Val.append(y_val)
    y_Test.append(y_test)

X_Train = np.array(X_Train).reshape(25*7,8)
X_Val = np.array(X_Val).reshape(7*7,8)
X_Test = np.array(X_Test).reshape(12*7,8)

y_Train = np.array(y_Train).reshape(25*7)
y_Val = np.array(y_Val).reshape(7*7)
y_Test = np.array(y_Test).reshape(12*7)




model = Sequential([
    LSTM(units=8, activation='relu', input_shape=(n_steps, n_features)),
    Dense(units=1)
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss='mse')

# Train the model
history = model.fit(X_Train, y_Train, epochs=100, batch_size=16, verbose=0,validation_data=(X_Val, y_Val))
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])

# Evaluate the model
mse = model.evaluate(X_Test, y_Test, verbose=0)
#print("Mean Squared Error on Test Data:", mse)


# Make predictions
predictions = model.predict(X_Test)
# Inverse scaling of predictions
predictions_inv = scaler.inverse_transform(predictions)


# Print predictions
print(predictions_inv)
print(scaler.inverse_transform(y_Test.reshape(12*7,1)))

for i in range(7):
    plt.figure()
    plt.plot(predictions_inv[i*12:(i+1)*12])
    plt.plot(scaler.inverse_transform(y_Test.reshape(12*7,1))[i*12:(i+1)*12])
    plt.xlabel(name[i])

test_y_predicted = np.array(predictions_inv).reshape(1,7,12)
np.save('test_y_predicted_lstm2.npy',test_y_predicted)



