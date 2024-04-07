import sys
sys.path.insert(0, 'code')

from dataset import AirportDataset, AirportGraph
from model import DeepAir
from train import train_model
import torch.nn as nn
import torch

if __name__ == '__main__':
    dataset = AirportDataset(training_start_quarter=[1,1998], 
                             batch_length=8, 
                             n_train_samples=24, 
                             n_test_samples=24, 
                             prediction_horizon=4)

    graph = AirportGraph(nodes=dataset.train_x_nodes,
                         edges=dataset.train_x_edges,
                         edge_attr=dataset.train_x_edges_attr,
                         y=dataset.train_y)

    graph.process()

    model = DeepAir(num_airports=len(dataset.qualified_ca_airports),
                    prediction_horizon=dataset.prediction_horizon)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = nn.MSELoss()
    num_epochs = 100

    train_model(model, optimizer, loss, num_epochs, graph)


    