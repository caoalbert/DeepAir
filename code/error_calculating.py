# Read .npy files and calculate the error between the predicted and actual values.
import numpy as np
import matplotlib.pyplot as plt


test_y_predicted_joey = np.load('../test_y_predicted_joey.npy')
test_y_joey = np.load('../test_y_joey.npy')

# print(test_y_predicted_joey.shape)
# print(test_y_joey.shape)

print(test_y_predicted_joey)
print(test_y_joey)

# Make test_y_predicted the same shape as test_y
test_y_predicted_joey = test_y_predicted_joey.reshape(test_y_joey.shape)


# Plot the predicted and actual values in subplots
fig, axs = plt.subplots(5, 1, figsize=(12, 17))
quarter = np.arange(1, 5, 1)
# fig.tight_layout()
for i in range(5):
    axs[i].plot(quarter, test_y_predicted_joey[0][i], label='Predicted')
    axs[i].plot(quarter, test_y_joey[0][i], label='Actual')
    axs[i].scatter(quarter, test_y_predicted_joey[0][i], color='red')
    axs[i].scatter(quarter, test_y_joey[0][i], color='red')
    axs[i].set_xlabel('Quarter')
    axs[i].set_ylabel('Throughput')
    axs[i].set_title('Predicted vs Actual Throughput')
    axs[i].legend()

plt.show()



