#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:10:26 2024

@author: jing
"""

# Read .npy files and calculate the error between the predicted and actual values.
import numpy as np
import matplotlib.pyplot as plt


test_y_predicted_joey = np.load('../test_y_predicted_jing.npy')

test_y_joey = np.load('../test_y_jing.npy')

# print(test_y_predicted_joey.shape)
# print(test_y_joey.shape)

print(test_y_predicted_joey)
print(test_y_joey)

# Make test_y_predicted the same shape as test_y
test_y_predicted_joey = test_y_predicted_joey.reshape(test_y_joey.shape)


# Plot the predicted and actual values in subplots
fig, axs = plt.subplots(11, 1, figsize=(6, 17))
quarter = np.arange(1, 5, 1)
# fig.tight_layout()
for i in range(11):
    axs[i].plot(quarter, test_y_predicted_joey[0][i], label='Predicted')
    #axs[i].plot(quarter, test_y_predicted_joey2[0][i], label='Predicted2')
    axs[i].plot(quarter, test_y_joey[0][i], label='Actual')
    axs[i].scatter(quarter, test_y_predicted_joey[0][i], color='red')
    axs[i].scatter(quarter, test_y_joey[0][i], color='red')
    #axs[i].scatter(quarter, test_y_predicted_joey2[0][i], color='red')
    axs[i].set_xlabel('Quarter')
    axs[i].set_ylabel('Throughput')
    axs[i].set_title('Predicted vs Actual Throughput')
    axs[i].legend()

plt.show()



