import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def train_loss_plot(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()


def loss_plot(train_loss, test_loss):
    # Plotting training and validation losses in 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    axs[0].plot(train_loss, label='Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')
    axs[0].legend()

    axs[1].plot(test_loss, label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')
    axs[1].legend()

    plt.show()




