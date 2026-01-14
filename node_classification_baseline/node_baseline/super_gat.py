import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F


class SuperGAT(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers, hidden_dim, num_heads, dropout, *args,
                 **kwargs):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(tnn.SuperGATConv(in_channel, hidden_dim, heads=num_heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_channel = hidden_dim
        self.convs.append(tnn.SuperGATConv(in_channel, out_channel, heads=num_heads, concat=False))
        self.bns.append(nn.BatchNorm1d(out_channel))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.convs[-1](x, edge_index)

        xs.append(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
