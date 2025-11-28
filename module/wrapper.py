import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap


class ModelGraph(nn.Module):
    def __init__(self, model_constructor, layers, hiddens, num_classes, model_name):
        super(ModelGraph, self).__init__()
        self.model_cons = model_constructor
        self.layers = layers
        self.hiddens = hiddens
        self.num_classes = num_classes
        self.model_name = model_name

    def reset_model(self):
        self.model = self.model_cons()
        self.linear = nn.Sequential(
            nn.Linear(self.layers * self.hiddens, self.hiddens * 2),
            nn.BatchNorm1d(2 * self.hiddens),
            nn.ReLU(True),
            nn.Linear(self.hiddens * 2, self.num_classes)
        )

    def forward(self, data):
        if self.model_name == 'EAT_MoE':
            h_node_list, batch, load_balance_loss = self.model(data)
            h_graph = gap(torch.cat(h_node_list, dim=-1), batch)
            # h_graph = gap(h_node_list[-1],batch)
            h_graph = self.linear(h_graph)
            return h_graph, load_balance_loss
        else:
            h_node_list, batch = self.model(data)
            h_graph = gap(torch.cat(h_node_list, dim=-1), batch)
            # h_graph = gap(h_node_list[-1],batch)
            h_graph = self.linear(h_graph)
            return h_graph


class ModelNode(nn.Module):
    def __init__(self, model_constructor, model_name):
        super(ModelNode, self).__init__()
        self.model_cons = model_constructor
        self.model_name = model_name

    def reset_model(self):
        self.model = self.model_cons()

    def forward(self, data):
        if self.model_name == 'EAT_MoE':
            h_node_list, _, load_balance_loss = self.model(data)
            return h_node_list[-1], load_balance_loss
        else:
            h_node_list, _ = self.model(data)
            return h_node_list[-1]
