import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class CATI_Net(nn.Module):
    def __init__(self, nsample, nfeat, nhid, nclass, dropout, alpha, nheads, WL):
        """CAT using implicit attention strategy"""
        super(CATI_Net, self).__init__()
        self.dropout = dropout

        self.sfeat = nn.Parameter(torch.FloatTensor(size=(nsample, nclass)))
        nn.init.xavier_normal_(self.sfeat.data, gain=1.414)

        # 2 layer network structure
        self.attentions = [
            CATIConv(nfeat, nhid, nsample, negative_slope=alpha, attn_drop=dropout, WL=WL) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = CATIConv(nhid * nheads, nclass, nsample, negative_slope=alpha, attn_drop=dropout,
                                WL=WL, concat=False)

    def forward(self, graph, x):
        # x is input feature matrix, graph is a dgl graph.

        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.cat([att(graph, x, self.sfeat) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)

        x = F.elu(self.out_att(graph, x, self.sfeat))

        return F.log_softmax(x, dim=1), self.sfeat


class CATIConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            n_samples,
            num_heads=1,
            feat_drop=0.0,
            attn_drop=0.0,
            wl_drop=0.9,
            negative_slope=0.2,
            residual=False,
            allow_zero_in_degree=False,
            norm="none",
            autobalance=True,
            WL=False,
            concat=True
    ):
        super(CATIConv, self).__init__()

        if norm not in ("none", "both"):
            raise DGLError('Invalid norm value. Must be either "none", "both".' ' But got "{}".'.format(norm))
        self._num_heads = num_heads

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        self.concat = concat
        self.WL = WL

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.WL_drop = nn.Dropout(wl_drop)

        # setting for leaky_relu
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if autobalance:
            self.aw = nn.Parameter(torch.zeros(size=(1, 2)))
            self.autobalance = True
        else:
            self.fw = 0.75
            self.autobalance = False

        if self.WL:
            self.epsilon = nn.Parameter(torch.zeros(size=(1, 1)))

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()

    def reset_parameters(self):
        # gain = nn.init.calculate_gain("relu")

        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)
        nn.init.xavier_normal_(self.aw.data, gain=1.414)
        nn.init.xavier_uniform_(self.aw.data, gain=1.414)
        # learnable parameter for WL
        if self.WL:
            nn.init.xavier_normal_(self.epsilon.data, gain=1.414)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=1.414)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, sfeat):

        sfeat_src = sfeat_dst = sfeat

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src, feat_dst = h_src, h_dst
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # print(self.epsilon)
            # if self.epsilon > 0:
            #     perm = self.epsilon / (graph.in_degrees() + 1e-9)
            #     perm = perm.unsqueeze(dim=1).unsqueeze(dim=1)
                # print(perm.size())
                # permfeat = feat_src * self.perm
                # print(permfeat.size())

            if self.WL:
                in_degree = (graph.in_degrees() + 1e-9).unsqueeze(dim=1).unsqueeze(dim=1)
                #  tanh function
                # perm = torch.tanh(self.epsilon) / in_degree
                # logistic function
                perm = 1 / (torch.exp(-self.epsilon) + 1)
                perm = perm / in_degree
                permfeat = self.WL_drop(feat_src * perm)

            # print(feat_src)
            # print(feat_dst)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # print(graph.srcdata['ft'])

            # Structural features
            graph.srcdata.update({"sft": sfeat_src})
            graph.dstdata.update({"dsft": sfeat_dst})
            # print(graph.srcdata['sft'])
            # print(graph.dstdata['dsft'])
            graph.apply_edges(fn.u_mul_v("sft", "dsft", "se"))

            # Structural attention
            se = graph.edata.pop("se")
            # print(se)
            sse = torch.sum(se, dim=1)

            sse = sse.unsqueeze(dim=1)
            sse = sse.unsqueeze(dim=1)

            ase = edge_softmax(graph, sse)
            # ase = ase.unsqueeze(dim=1)
            # ase = ase.unsqueeze(dim=1)

            # print(ase.size())
            #
            # print(ase)

            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            # compute softmax
            e = edge_softmax(graph, e)

            # for CAT-I
            if self.autobalance:
                weight = nn.functional.softmax(self.aw, dim=1)
                e = weight[0, 0] * e + weight[0, 1] * ase
            else:
                e = self.fw * e + (1 - self.fw) * ase

            # construct WL filter
            # if self.WL:
            #     graph.edata['a'] = e
            #     graph.edges[self.idx, self.idx].data['a'] = graph.edges[self.idx, self.idx].data['a'] + perm
            #     e = graph.edata.pop("a")

            graph.edata['a'] = self.attn_drop(e)
            # print(graph.edata['a'].size())
            # graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # print(rst.size())
            # if self.epsilon > 0:
            #     rst = rst + permfeat
            if self.WL:
                rst = rst + permfeat
            # print(feat_src)
            # print(feat_dst)
            # print(graph.dstdata["ft"])

            if self._norm == "both":
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self.concat:
                rst = F.elu(torch.flatten(rst, start_dim=1))
                return rst
            else:
                rst = torch.flatten(rst, start_dim=1)
                return rst
