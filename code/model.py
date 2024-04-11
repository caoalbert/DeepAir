import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeGATConv


class DeepAir(nn.Module):
    def __init__(self, num_airports, prediction_horizon):
        super(DeepAir, self).__init__()
        self.gat = EdgeGATConv(in_feats=1,
                               edge_feats=1,
                               out_feats=8,
                               num_heads=1)

        self.lstm = nn.LSTM(8, 24, num_layers=1, batch_first=True)
        
        self.gru = nn.GRU(8, 24, batch_first=True)
        self.fc = nn.Linear(24, num_airports * prediction_horizon)

        self.dropout = nn.Dropout(0.1)
        
        self.fcc = nn.Linear(8*20, 20)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, batched_data_series):
        batched_gru_inputs = []
        for data_series in batched_data_series:  
            gru_inputs = []
            for graph in data_series: 
                feature = self.gat(graph, graph.ndata["x"], graph.edata["weight"])
                feature = feature.view(feature.size(0), -1)  # Reshape to concatenate head features


                
                graph_feature = feature.mean(dim=0, keepdim=True)  # Shape: [1, num_heads * out_feats]
                
                gru_inputs.append(graph_feature)

            gru_inputs = torch.stack(gru_inputs, dim=1) 
            batched_gru_inputs.append(gru_inputs)


        all_series = torch.cat(batched_gru_inputs, dim=0)  # Shape: [batch_size, seq_length, num_heads * out_feats]
        # _, h_n = self.gru(all_series)  # Process all series through the GRU
        all_series = self.dropout(all_series)
        outputs, hidden = self.lstm(all_series)  # Process all series through the LSTM
        out = self.fc(outputs.squeeze(0))  # Get final output
        out = self.fcc(out.view(-1, 8*20))
        return out