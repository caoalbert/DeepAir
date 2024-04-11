import sys

sys.path.insert(0, 'code')

from dataset_normalized import AirportDataset, AirportGraph
from model import DeepAir
from train import train_model, train_model_plot
from visualization import loss_plot
import torch.nn as nn
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = 'trained_models/albert0408.pth'
PARAMS_PATH = f'trained_models/albert0408.pth'

MODEL_PATH_Joey = 'trained_models/joey0410.pth'
PARAMS_PATH_Joey = f'trained_models/joey0410.pth'

if __name__ == '__main__':
    dataset = AirportDataset(training_start_quarter=[1, 1998],
                             batch_length=8,
                             n_train_samples=60,
                             n_test_samples=1,
                             prediction_horizon=4)

    training_graph = AirportGraph(nodes=dataset.train_x_nodes,
                                  edges=dataset.train_x_edges,
                                  edge_attr=dataset.train_x_edges_attr,
                                  y=dataset.train_y)

    testing_graph = AirportGraph(nodes=dataset.test_x_nodes,
                                 edges=dataset.test_x_edges,
                                 edge_attr=dataset.test_x_edges_attr,
                                 y=dataset.test_y)

    training_graph.process()
    testing_graph.process()

    model = DeepAir(num_airports=len(dataset.qualified_ca_airports),
                    prediction_horizon=dataset.prediction_horizon)
    model_params = (len(dataset.qualified_ca_airports), dataset.prediction_horizon)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = nn.MSELoss()
    num_epochs = 200

    train_losses, test_losses = train_model_plot(model, optimizer, loss, num_epochs, training_graph, testing_graph)

    torch.save(model.state_dict(), MODEL_PATH_Joey)

    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump((len(dataset.qualified_ca_airports), dataset.prediction_horizon), f)

    # Plotting training and validation losses
    loss_plot(train_losses, test_losses)

    # Model evaluation as before
    model.eval()
    with torch.no_grad():
        test_y_predicted = model(testing_graph.series).detach().numpy()
        test_y_predicted = np.exp(test_y_predicted) - 1  # Reversing log transformation

    np.save('test_y_predicted_joey.npy', test_y_predicted)
    np.save('test_y_joey.npy', dataset.test_y)




