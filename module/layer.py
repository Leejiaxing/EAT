import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap, global_max_pool as gxp
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GCNSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GCNSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(sublayers - 1):
            self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=subhiddens))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        xs = []
        for i in range(self.sublayers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)
        x = gmp(torch.cat(xs, dim=-1), node_to_subgraph)
        return x


class GINSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GINSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )

        for _ in range(sublayers - 1):
            self.sub_gnns.append(
                tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=subhiddens), train_eps=True))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_gnns.append(
            tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=out_channels), train_eps=True))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph=None):

        xs = []
        for i in range(self.sublayers):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)
        x = gmp(torch.cat(xs, dim=-1), node_to_subgraph)
        # x = gap(xs[-1], node_to_subgraph)
        return x


class GINSublayer_VN(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(GINSublayer_VN, self).__init__()
        self.sublayers = sublayers
        self.sub_gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        def mlp(inchannel, hidden, outchannel):
            return torch.nn.Sequential(
                torch.nn.Linear(inchannel, hidden),
                torch.nn.BatchNorm1d(hidden),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden, outchannel),
            )

        self.vn_ebd = torch.nn.Embedding(1, subhiddens)
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(sublayers - 1):
            self.sub_gnns.append(
                tnn.GINConv(mlp(inchannel=in_channels, hidden=subhiddens, outchannel=subhiddens), train_eps=True))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(subhiddens, 2 * subhiddens), torch.nn.BatchNorm1d(2 * subhiddens), torch.nn.ReLU(),
                torch.nn.Linear(2 * subhiddens, subhiddens), torch.nn.BatchNorm1d(subhiddens), torch.nn.ReLU()))

        self.sub_gnns.append(tnn.GCNConv(in_channels=in_channels, out_channels=out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.vn_ebd.weight.data, 0)
        for layer in self.sub_gnns:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        vne = self.vn_ebd(
            torch.zeros(node_to_subgraph[-1].item() + 1, dtype=sub_edge_index.dtype, device=sub_edge_index.device))
        # x = x + vne[node_to_subgraph]
        xs = []
        for i in range(self.sublayers - 1):
            x = self.sub_gnns[i](x, sub_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)

            vnet = gap(xs[i], node_to_subgraph) + vne
            vne = F.dropout(self.mlp_virtualnode_list[i](vnet), 0.5, training=self.training)
            x = x + vne[node_to_subgraph]

        xs[-1] = xs[-1] + vne[node_to_subgraph]
        x = F.dropout(self.bns[-1](self.sub_gnns[-1](xs[-1], sub_edge_index)), 0.5, self.training)
        xs.append(x)

        node_p = 0
        for layer in range(self.sublayers):
            node_p += xs[layer]
        x = gmp(node_p, node_to_subgraph)
        return x


class MLPSublayer(nn.Module):
    def __init__(self, in_channels, out_channels, sublayers, subhiddens):
        super(MLPSublayer, self).__init__()
        self.sublayers = sublayers
        self.sub_lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(sublayers - 1):
            self.sub_lins.append(nn.Linear(in_channels, subhiddens))
            self.bns.append(nn.BatchNorm1d(subhiddens))
            in_channels = subhiddens

        self.sub_lins.append(nn.Linear(in_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.sub_lins:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, sub_edge_index, node_to_subgraph):

        xs = []
        for i in range(self.sublayers):
            x = self.sub_lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, 0.5, self.training)
            xs.append(x)
        x = gmp(torch.cat(xs, dim=-1), node_to_subgraph)
        return x
