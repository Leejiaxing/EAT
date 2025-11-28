import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, global_max_pool as gxp
from module.moe import MoE


class GIN_Sublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sub_hidden_dim, num_sub_layers):
        super(GIN_Sublayer, self).__init__()
        self.num_sub_layers = num_sub_layers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        def mlp(in_channel, hidden, out_channel):
            return torch.nn.Sequential(
                torch.nn.Linear(in_channel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, out_channel),
            )

        for _ in range(num_sub_layers - 1):
            self.sub_gnns.append(
                tnn.GINConv(mlp(in_channel=in_channels, hidden=sub_hidden_dim, out_channel=sub_hidden_dim),
                            train_eps=True))
            self.bns.append(nn.BatchNorm1d(sub_hidden_dim))
            in_channels = sub_hidden_dim

        self.sub_gnns.append(
            tnn.GINConv(mlp(in_channel=in_channels, hidden=sub_hidden_dim, out_channel=out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index):
        xs = 0
        for i in range(self.num_sub_layers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs += x

        return xs


class GCN_Sublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sub_hidden_dim, num_sub_layers):
        super(GCN_Sublayer, self).__init__()
        self.num_sub_layers = num_sub_layers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_sub_layers - 1):
            self.sub_gnns.append(
                tnn.GCNConv(in_channels=in_channels, out_channels=sub_hidden_dim))
            self.bns.append(nn.BatchNorm1d(sub_hidden_dim))
            in_channels = sub_hidden_dim

        self.sub_gnns.append(
            tnn.GCNConv(in_channels=in_channels, out_channels=out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index):

        xs = []
        for i in range(self.num_sub_layers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)

        return x


class SAGE_Sublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sub_hidden_dim, num_sub_layers):
        super(SAGE_Sublayer, self).__init__()
        self.num_sub_layers = num_sub_layers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_sub_layers - 1):
            self.sub_gnns.append(
                tnn.SAGEConv(in_channels=in_channels, out_channels=sub_hidden_dim))
            self.bns.append(nn.BatchNorm1d(sub_hidden_dim))
            in_channels = sub_hidden_dim

        self.sub_gnns.append(
            tnn.SAGEConv(in_channels=in_channels, out_channels=out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index):

        xs = []
        for i in range(self.num_sub_layers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)

        return x


class MLP_Sublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sub_hidden_dim, num_sub_layers):
        super(MLP_Sublayer, self).__init__()
        self.num_sub_layers = num_sub_layers
        self.sub_lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_sub_layers - 1):
            self.sub_lins.append(nn.Linear(in_channels, sub_hidden_dim))
            self.bns.append(nn.BatchNorm1d(sub_hidden_dim))
            in_channels = sub_hidden_dim

        self.sub_lins.append(nn.Linear(in_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_lins:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index):

        xs = []
        for i in range(self.num_sub_layers):
            x = self.sub_lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)

        return x


class GNN_SpMoE_subgraph(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, in_channel, hidden_dim, num_experts=6, gnn_type='sage', k=3, coef=1e-2):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
            k: k value for top-k sparse gating. 
            num_experts: total number of experts in each layer. 
            num_experts_1hop: number of hop-1 experts in each layer. The first num_experts_1hop are hop-1 experts. The rest num_experts-num_experts_1hop are hop-2 experts.
        '''

        super(GNN_SpMoE_subgraph, self).__init__()
        self.load_balance_loss = 0
        self.num_experts = num_experts
        self.k = k

        convs_list = torch.nn.ModuleList()
        for expert_idx in range(2):
            if gnn_type == 'mlp':
                convs_list.append(MLP_Sublayer(in_channel, hidden_dim, num_sub_layers=1, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gin':
                convs_list.append(GIN_Sublayer(in_channel, hidden_dim, num_sub_layers=1, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gcn':
                convs_list.append(GCN_Sublayer(in_channel, hidden_dim, num_sub_layers=1, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'sage':
                convs_list.append(SAGE_Sublayer(in_channel, hidden_dim, num_sub_layers=1, sub_hidden_dim=hidden_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
        for expert_idx in range(2):
            if gnn_type == 'mlp':
                convs_list.append(MLP_Sublayer(in_channel, hidden_dim, num_sub_layers=2, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gin':
                convs_list.append(
                    GIN_Sublayer(in_channel, hidden_dim, num_sub_layers=2, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gcn':
                convs_list.append(
                    GCN_Sublayer(in_channel, hidden_dim, num_sub_layers=2, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'sage':
                convs_list.append(SAGE_Sublayer(in_channel, hidden_dim, num_sub_layers=3, sub_hidden_dim=hidden_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
        for expert_idx in range(2):
            if gnn_type == 'mlp':
                convs_list.append(MLP_Sublayer(in_channel, hidden_dim, num_sub_layers=3, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gin':
                convs_list.append(
                    GIN_Sublayer(in_channel, hidden_dim, num_sub_layers=3, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'gcn':
                convs_list.append(
                    GCN_Sublayer(in_channel, hidden_dim, num_sub_layers=3, sub_hidden_dim=hidden_dim))
            elif gnn_type == 'sage':
                convs_list.append(SAGE_Sublayer(in_channel, hidden_dim, num_sub_layers=3, sub_hidden_dim=hidden_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

        self.ffn = MoE(input_size=in_channel, output_size=hidden_dim, num_experts=num_experts, experts_conv=convs_list, k=k, coef=coef)

    def forward(self, x, sub_edge_index, node_to_subgraph):
        h, load_balance_loss = self.ffn(x, sub_edge_index)
        x = gmp(h, node_to_subgraph)

        return x, load_balance_loss
