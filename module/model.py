from tkinter.tix import Tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from module.layer import GCNSublayer, GINSublayer, MLPSublayer, GINSublayer_VN
from torch_geometric.nn import GCNConv, GATConv
from module.conv import GNN_SpMoE_subgraph


def get_subx(node_x, sub_x_idx, subgraph_to_graph, batch):
    batch_size = batch.max() + 1
    sub_x = []
    for i in range(batch_size):
        g_nodes = node_x[subgraph_to_graph == i]
        sub_idx = sub_x_idx[batch == i]
        assert g_nodes.size(0) == sub_idx.max().item() + 1
        sub_x.append(g_nodes[sub_idx])
    return torch.cat(sub_x, dim=0)


class EAT(nn.Module):
    def __init__(self, num_features, out_channels, num_layers, num_sub_layers, num_heads, use_z, use_rd, use_pe, hidden_dim,
                 dataset_name, dropout):
        super(EAT, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.use_pe = use_pe
        self.layers = num_layers
        self.dropout = dropout

        if dataset_name == "ZINC":
            self.node_embedding = nn.Embedding(num_features, hidden_dim)
        elif "mol" in dataset_name:
            self.node_embedding = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Linear(num_features, hidden_dim)
        in_channel = hidden_dim

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
            in_channel += 8
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
            in_channel += 8
        if self.use_pe:
            self.pe_projection = torch.nn.Linear(4, 8)
            in_channel += 8

        self.sub_convs = nn.ModuleList()
        self.att_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.sub_convs.append(GCNSublayer(in_channel, hidden_dim, sublayers=num_sub_layers, subhiddens=hidden_dim))
            self.att_convs.append(GATConv(hidden_dim * num_sub_layers, hidden_dim, heads=num_heads, concat=False,
                                          dropout=self.dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_channel = hidden_dim
        self.sub_convs.append(GCNSublayer(in_channel, hidden_dim, sublayers=num_sub_layers, subhiddens=hidden_dim))
        self.att_convs.append(
            GATConv(hidden_dim * num_sub_layers, out_channels, heads=num_heads, concat=False, dropout=self.dropout))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # self.node_embedding.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        if self.use_pe:
            self.pe_projection.reset_parameters()
        for l in range(self.layers):
            self.sub_convs[l].reset_parameters()
            self.att_convs[l].reset_parameters()

    def forward(self, data):
        edge_index, x = data.original_edge_index, data.original_x
        sub_x_index, sub_edge_idx = data.node_index, data.edge_index
        node_to_subgraph, subgraph_to_graph = data.node_to_subgraph, data.subgraph_to_graph

        addition_emb = []
        if self.use_z and 'z' in data:
            z = self.z_embedding(data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            addition_emb.append(z)

        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            addition_emb.append(rd_proj)

        if self.use_pe:
            pe_proj = self.pe_projection(data.lpe_)
            addition_emb.append(pe_proj)
        addition_emb = torch.cat(addition_emb, dim=-1)

        x = self.node_embedding(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index]
        xs = []
        if self.use_z or self.use_rd:
            sub_x = torch.cat([addition_emb, sub_x], -1)

        sub_x = F.dropout(sub_x, p=self.dropout, training=self.training, inplace=True)
        for i in range(self.layers - 1):
            x = self.sub_convs[i](sub_x, sub_edge_idx, node_to_subgraph)
            x = self.att_convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

            sub_x = x[sub_x_index]

        x = self.sub_convs[-1](sub_x, sub_edge_idx, node_to_subgraph)
        x = self.att_convs[-1](x, edge_index)
        # x = self.bns[-1](x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        return xs, subgraph_to_graph


class EAT_MoE(nn.Module):
    def __init__(self, num_features, out_channels, num_layers, num_heads, use_z, use_rd, use_pe, hidden_dim,
                 dataset_name, dropout, sub_encoder):
        super(EAT_MoE, self).__init__()
        self.use_rd = use_rd
        self.use_z = use_z
        self.use_pe = use_pe
        self.layers = num_layers
        self.dropout = dropout
        self.sub_encoder = sub_encoder

        if dataset_name == "ZINC":
            self.node_embedding = nn.Embedding(num_features, hidden_dim)
        elif "mol" in dataset_name:
            self.node_embedding = AtomEncoder(hidden_dim)
        else:
            self.node_embedding = nn.Linear(num_features, hidden_dim)
        in_channel = hidden_dim

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
            in_channel += 8
        if self.use_z:
            self.z_embedding = torch.nn.Embedding(1000, 8)
            in_channel += 8
        if self.use_pe:
            self.pe_projection = torch.nn.Linear(4, 8)
            in_channel += 8

        self.sub_convs = nn.ModuleList()
        self.att_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.sub_convs.append(GNN_SpMoE_subgraph(in_channel=in_channel, hidden_dim=hidden_dim, gnn_type=self.sub_encoder))
            self.att_convs.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                                          dropout=self.dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_channel = hidden_dim
        self.sub_convs.append(GNN_SpMoE_subgraph(in_channel=in_channel, hidden_dim=hidden_dim, gnn_type=self.sub_encoder))
        self.att_convs.append(
            GATConv(hidden_dim, out_channels, heads=num_heads, concat=False, dropout=self.dropout))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        # self.node_embedding.reset_parameters()
        if self.use_z:
            self.z_embedding.reset_parameters()
        if self.use_pe:
            self.pe_projection.reset_parameters()
        for l in range(self.layers):
            self.att_convs[l].reset_parameters()

    def forward(self, data):
        edge_index, x = data.original_edge_index, data.original_x
        sub_x_index, sub_edge_idx = data.node_index, data.edge_index
        node_to_subgraph, subgraph_to_graph = data.node_to_subgraph, data.subgraph_to_graph

        addition_emb = []
        if self.use_z and 'z' in data:
            z = self.z_embedding(data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            addition_emb.append(z)

        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            addition_emb.append(rd_proj)

        if self.use_pe:
            pe_proj = self.pe_projection(data.lpe_)
            addition_emb.append(pe_proj)
        addition_emb = torch.cat(addition_emb, dim=-1)

        x = self.node_embedding(x)
        if len(x.shape) == 3:
            x = torch.sum(x, dim=-2)
        sub_x = x[sub_x_index]
        xs = []
        if self.use_z or self.use_rd:
            sub_x = torch.cat([addition_emb, sub_x], -1)

        load_balance_losses = 0
        sub_x = F.dropout(sub_x, p=self.dropout, training=self.training, inplace=True)
        for i in range(self.layers - 1):
            x, load_balance_loss = self.sub_convs[i](sub_x, sub_edge_idx, node_to_subgraph)
            x = self.att_convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

            sub_x = x[sub_x_index]
            load_balance_losses += load_balance_loss

        x, load_balance_loss = self.sub_convs[-1](sub_x, sub_edge_idx, node_to_subgraph)
        x = self.att_convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        load_balance_losses += load_balance_loss

        return xs, subgraph_to_graph, load_balance_losses / self.layers

