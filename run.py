import sys
sys.path.insert(0, 'code')

from dataset import AirportDataset, AirportGraph
from model import DeepAir
from train import train_model
import torch.nn as nn
import torch
import pickle
import numpy as np
import os

MODEL_PATH = 'trained_models/albert0408.pth'
PARAMS_PATH = f'trained_models/albert0408.pth'

if __name__ == '__main__':
    dataset = AirportDataset(training_start_quarter=[1,2003], 
                             batch_length=8, 
                             n_train_samples=36, 
                             n_test_samples=1, 
                             prediction_horizon=4)

    graph = AirportGraph(nodes=dataset.train_x_nodes,
                         edges=dataset.train_x_edges,
                         edge_attr=dataset.train_x_edges_attr,
                         y=dataset.train_y)

    graph.process()

    model = DeepAir(num_airports=len(dataset.qualified_ca_airports),
                    prediction_horizon=dataset.prediction_horizon)
    model_params = (len(dataset.qualified_ca_airports), dataset.prediction_horizon)
    y_scaler = dataset.y_scaler

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = nn.MSELoss()
    num_epochs = 200

    train_model(model, optimizer, loss, num_epochs, graph)

    torch.save(model.state_dict(), MODEL_PATH)

    with open(PARAMS_PATH, 'wb') as f:
        pickle.dump(model_params, f)

    model.eval()
    test_graph = AirportGraph(nodes=dataset.test_x_nodes,
                              edges=dataset.test_x_edges,
                              edge_attr=dataset.test_x_edges_attr,
                              y=dataset.test_y)
    test_graph.process()
    test_y_predicted = model(test_graph.series)
    test_y_predicted = test_y_predicted.detach().numpy().reshape(-1, dataset.prediction_horizon)
    test_y_predicted = y_scaler.inverse_transform(test_y_predicted)

    train_y_predicted = model(graph.series)
    train_y_predicted = train_y_predicted.detach().numpy().reshape(-1, dataset.prediction_horizon)
    train_y_predicted = y_scaler.inverse_transform(train_y_predicted)

    if os.path.exists('train_y_predicted.npy'):
        os.remove('train_y_predicted.npy')
    if os.path.exists('train_y.npy'):
        os.remove('train_y.npy')
    if os.path.exists('test_y_predicted.npy'):
        os.remove('test_y_predicted.npy')
    if os.path.exists('test_y.npy'):
        os.remove('test_y.npy')


    train_y = dataset.train_y
    for i in range(len(train_y)):
        train_y[i] = y_scaler.inverse_transform(train_y[i].reshape(-1, dataset.prediction_horizon))

    test_y = dataset.test_y
    for i in range(len(test_y)):
        test_y[i] = y_scaler.inverse_transform(test_y[i].reshape(-1, dataset.prediction_horizon))

    np.save('train_y_predicted.npy', train_y_predicted)
    np.save('train_y.npy', np.array(train_y))

    np.save('test_y_predicted.npy',test_y_predicted)
    np.save('test_y.npy', np.array(test_y))



    