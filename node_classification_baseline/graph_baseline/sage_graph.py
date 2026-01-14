import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
from ogb.graphproppred.mol_encoder import AtomEncoder


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_dim, dropout, dataset_name):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if dataset_name == "ZINC":
            self.node_embedding = nn.Embedding(num_features, self.hidden_dim)
        elif "mol" in dataset_name:
            self.node_embedding = AtomEncoder(self.hidden_dim)
        else:
            self.node_embedding = nn.Linear(num_features, self.hidden_dim)
        in_channel = num_features
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(tnn.SAGEConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            in_channel = hidden_dim
        self.convs.append(tnn.SAGEConv(in_channel, self.hidden_dim))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.linear = nn.Sequential(
            nn.Linear(self.num_layers * self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(2 * self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim * 2, self.num_classes)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # self.node_embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_embedding(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        xs.append(x)
        h_graph = gap(torch.cat(xs, dim=-1), batch)
        h_graph = self.linear(h_graph)

        return h_graph
