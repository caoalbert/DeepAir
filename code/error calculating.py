# Read .npy files and calculate the error between the predicted and actual values.
import numpy as np
import matplotlib.pyplot as plt


test_y_predicted_joey = np.load('../test_y_predicted_joey.npy')
test_y_joey = np.load('../test_y_joey.npy')

# print(test_y_predicted_joey.shape)
# print(test_y_joey.shape)

# print(test_y_predicted_joey)
# print(test_y_joey)

# Make test_y_predicted the same shape as test_y
test_y_predicted_joey = test_y_predicted_joey.reshape(test_y_joey.shape)


# Plot the predicted and actual values
plt.figure(figsize=(10, 6))
plt.plot(test_y_predicted_joey[0], label='Predicted')
plt.plot(test_y_joey[0], label='Actual')
plt.xlabel('Quarter')
plt.ylabel('Throughput')
plt.title('Predicted vs Actual Throughput')
plt.legend()
plt.show()
