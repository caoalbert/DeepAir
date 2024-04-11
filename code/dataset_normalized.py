
import torch
from torch.utils.data import TensorDataset
import data_preprocess
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import dgl


class AirportDataset(TensorDataset):
    def _convert_to_quarter_index(self, list_for_time):
        quarter = list_for_time[0]
        year = list_for_time[1]
        return (year - 1993) * 4 + quarter

    def _select_qualified_airports(self):
        # Select qualified airports in the throughput dataset
        # Select qualified CA airports based on the throughput dataset

        quarters_required_in_od = np.arange(self.training_x_start_quarter, self.testing_x_end_quarter)
        quarters_required_in_throughput = np.arange(self.training_x_start_quarter, self.testing_y_end_quarter)

        throughput_selected = self.throughput[self.throughput['quarter_id'].isin(quarters_required_in_throughput)]
        self.throughput_selected = throughput_selected

        throughput_selected_cnt = throughput_selected.groupby(['Facility'])['quarter_id'].nunique().reset_index()
        qualified_airports = \
        throughput_selected_cnt[throughput_selected_cnt['quarter_id'] >= len(quarters_required_in_throughput)][
            'Facility'].values

        qualified_ca_airports = set(qualified_airports).intersection(self.ca_airport)

        print('Number of airports in the network:', len(qualified_airports))
        print('The airports in the network are:', qualified_airports)
        print('Number of California airports in the network:', len(qualified_ca_airports))
        print('The Californian airports in the network are:', qualified_ca_airports)

        qualified_airports = {index: value for index, value in enumerate(qualified_airports)}
        qualified_ca_airports = {index: value for index, value in enumerate(qualified_ca_airports)}

        qualified_airports_inverse = {value: index for index, value in qualified_airports.items()}
        qualified_ca_airports_inverse = {value: index for index, value in qualified_ca_airports.items()}

        return qualified_airports, qualified_ca_airports, qualified_airports_inverse, qualified_ca_airports_inverse

    def build_xy(self,
                 x_start_quarter,
                 x_end_quarter,
                 y_start_quarter,
                 y_end_quarter,
                 qualified_airports,
                 qualified_airports_inverse,
                 qualified_ca_airports,
                 qualified_ca_airports_inverse
                 ):
        df = self.df
        throughput_selected = self.throughput_selected

        num_edges = len(qualified_ca_airports) * (len(qualified_airports) - 1) * 2

        x_nodes = np.empty((x_end_quarter - x_start_quarter, len(qualified_airports), 1), dtype=int)
        x_edges_all = np.empty((x_end_quarter - x_start_quarter, num_edges, 2), dtype=int)
        x_edge_attr_all = np.empty((x_end_quarter - x_start_quarter, num_edges, 1), dtype=int)
        y = np.empty((len(qualified_ca_airports), y_end_quarter - y_start_quarter))

        for q_id in range(x_start_quarter, x_end_quarter):
            x_edges = np.empty((1, 2), dtype=int)
            x_edge_attr = np.empty((1, 1), dtype=int)

            df_q_id = df[(df['quarter_id'] == q_id)]
            df_q_id = df_q_id[df_q_id['Origin'].isin(qualified_airports.values())]
            df_q_id = df_q_id[df_q_id['Dest'].isin(qualified_airports.values())].reset_index(drop=True)
            throughput_q_id = throughput_selected[throughput_selected['quarter_id'] == q_id]

            for node_index in range(len(qualified_airports)):
                try:
                    x_nodes[q_id - x_start_quarter, node_index, 0] = \
                    throughput_q_id[throughput_q_id['Facility'] == qualified_airports[node_index]][
                        'total_throughput'].values[0]
                except IndexError:
                    x_nodes[q_id - x_start_quarter, node_index, 0] = 0

            for origin in qualified_ca_airports.values():
                for dest in qualified_airports.values():
                    if origin != dest:
                        x_edges = np.concatenate(
                            (x_edges, [[qualified_airports_inverse[origin], qualified_airports_inverse[dest]]]), axis=0)
                        try:
                            passengers = \
                            df_q_id[(df_q_id['Origin'] == origin) & (df_q_id['Dest'] == dest)]['Passengers'].values[0]
                            x_edge_attr = np.concatenate((x_edge_attr, [[passengers]]), axis=0)
                        except IndexError:
                            x_edge_attr = np.concatenate((x_edge_attr, [[0]]), axis=0)

            for origin in qualified_airports.values():
                for dest in qualified_ca_airports.values():
                    if origin != dest:
                        x_edges = np.concatenate(
                            (x_edges, [[qualified_airports_inverse[origin], qualified_airports_inverse[dest]]]), axis=0)
                        try:
                            passengers = \
                            df_q_id[(df_q_id['Origin'] == origin) & (df_q_id['Dest'] == dest)]['Passengers'].values[0]
                            x_edge_attr = np.concatenate((x_edge_attr, [[passengers]]), axis=0)
                        except IndexError:
                            x_edge_attr = np.concatenate((x_edge_attr, [[0]]), axis=0)

            x_edges = np.delete(x_edges, 0, axis=0)
            x_edge_attr = np.delete(x_edge_attr, 0, axis=0)

            x_edges_all[q_id - x_start_quarter] = x_edges
            x_edge_attr_all[q_id - x_start_quarter] = x_edge_attr

        for q_id in range(y_start_quarter, y_end_quarter):
            throughput_q_id = throughput_selected[throughput_selected['quarter_id'] == q_id]
            for ca_airport in qualified_ca_airports.values():
                throughput = throughput_q_id[throughput_q_id['Facility'] == ca_airport]['total_throughput'].values[0]
                y[qualified_ca_airports_inverse[ca_airport], q_id - y_start_quarter] = throughput

        return x_nodes, x_edges_all, x_edge_attr_all, y

    def _build_xxx_set(self, batch_length, n_samples, x_start, y_start, prediction_horizon):
        set_x_nodes, set_x_edges, set_x_edges_attr, set_y = [], [], [], []
        for i in tqdm(range(n_samples)):
            x_start_input = x_start + i
            x_end_input = x_start + i + batch_length
            y_start_input = y_start + i
            y_end_input = y_start + i + prediction_horizon
            x_nodes, x_edges, x_edges_attr, y = \
                self.build_xy(x_start_input,
                              x_end_input,
                              y_start_input,
                              y_end_input,
                              self.qualified_airports,
                              self.qualified_airports_inverse,
                              self.qualified_ca_airports,
                              self.qualified_ca_airports_inverse)

            set_x_nodes.append(x_nodes)
            set_x_edges.append(x_edges)
            set_x_edges_attr.append(x_edges_attr)
            set_y.append(y)

        return set_x_nodes, set_x_edges, set_x_edges_attr, set_y

    def __init__(
            self,
            training_start_quarter,
            batch_length,
            n_train_samples,
            n_test_samples,
            prediction_horizon
    ):
        # Load the throughput data
        self.prediction_horizon = prediction_horizon
        self.throughput = data_preprocess._pre_process_throughput()
        # Locate the aspm77 airports
        aspm77 = self.throughput['Facility'].unique()
        self.aspm77 = {index: value for index, value in enumerate(aspm77)}
        # Load the od demand data
        self.df, self.ca_airport, self.selected_airports = data_preprocess._pre_process_od(self.aspm77)

        # Initialize the scalers
        self.scaler_node_features = StandardScaler()
        self.scaler_edge_features = StandardScaler()

        # Calculate the corresponding quarter index for different sets
        self.training_x_start_quarter = self._convert_to_quarter_index(training_start_quarter)
        self.training_x_end_quarter = self.training_x_start_quarter + batch_length + n_train_samples
        self.training_y_start_quarter = self.training_x_start_quarter + batch_length
        self.training_y_end_quarter = self.training_y_start_quarter + prediction_horizon + n_train_samples

        self.testing_x_start_quarter = self.training_x_end_quarter
        self.testing_x_end_quarter = self.testing_x_start_quarter + batch_length + n_test_samples
        self.testing_y_start_quarter = self.testing_x_start_quarter + batch_length
        self.testing_y_end_quarter = self.testing_y_start_quarter + prediction_horizon + n_test_samples

        # Find qualified airports
        self.qualified_airports, self.qualified_ca_airports, self.qualified_airports_inverse, self.qualified_ca_airports_inverse = self._select_qualified_airports()

        # Build training set
        print('Building training set')
        train_x_nodes, train_x_edges, train_x_edges_attr, train_y = \
            self._build_xxx_set(batch_length, n_train_samples, self.training_x_start_quarter,
                                self.training_y_start_quarter, prediction_horizon)

        # Build the testing set
        print('Building testing set')
        test_x_nodes, test_x_edges, test_x_edges_attr, test_y = \
            self._build_xxx_set(batch_length, n_test_samples, self.testing_x_start_quarter,
                                self.testing_y_start_quarter, prediction_horizon)

        self.train_x_nodes, self.train_x_edges, self.train_x_edges_attr, self.train_y = train_x_nodes, train_x_edges, train_x_edges_attr, train_y

        # Normalize the features
        self.train_x_nodes, self.train_x_edges_attr = self.normalize_features(self.train_x_nodes, self.train_x_edges_attr)

        for i in range(len(self.train_y)):
            self.train_y[i] = np.log(self.train_y[i] + 1)

        self.test_x_nodes, self.test_x_edges, self.test_x_edges_attr, self.test_y = test_x_nodes, test_x_edges, test_x_edges_attr, test_y
        self.test_x_nodes, self.test_x_edges_attr = self.normalize_features(self.test_x_nodes, self.test_x_edges_attr)

    def get_ca_airport_index(self):
        return self.qualified_ca_airports_inverse

    def get_all_airport_index(self):
        return self.qualified_airports_inverse

    def normalize_features(self, x_nodes, x_edge_attr):
        # Convert lists to numpy arrays if they are not already
        x_nodes = np.array(x_nodes, dtype=np.float32)  # Ensure dtype is float for scaler
        x_edge_attr = np.array(x_edge_attr, dtype=np.float32)

        # Reshape for scaler, fit_transform, and then reshape back
        x_nodes = self.scaler_node_features.fit_transform(x_nodes.reshape(-1, 1)).reshape(x_nodes.shape)
        x_edge_attr = self.scaler_edge_features.fit_transform(x_edge_attr.reshape(-1, 1)).reshape(x_edge_attr.shape)

        return x_nodes, x_edge_attr

class AirportGraph():
    def __init__(self, nodes, edges, edge_attr, y):
        self.nodes = nodes
        self.edges = edges
        self.edge_attr = edge_attr
        self.y = y

    def process(self):
        self.series = []
        self.y_transformed = np.empty(shape=(len(self.y), self.y[0].shape[0] * self.y[0].shape[1]))

        for series in range(len(self.nodes)):
            node_sample = self.nodes[series]
            edge_sample = self.edges[series]
            edge_attr_sample = self.edge_attr[series]

            list_of_graphs = []

            for graph in range(node_sample.shape[0]):
                transformed_edges = edge_sample[graph].T
                g = dgl.graph(
                    (torch.LongTensor(transformed_edges[0]), torch.LongTensor(transformed_edges[1]))
                )

                g.ndata["x"] = torch.tensor(node_sample[graph]).float()
                g.edata["weight"] = torch.tensor(edge_attr_sample[graph]).float()

                list_of_graphs.append(g)

            self.series.append(list_of_graphs)

            self.y_transformed[series] = self.y[series].flatten()

        self.y_transformed = torch.tensor(self.y_transformed).float()














