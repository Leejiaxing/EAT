from typing import Union, Tuple, Optional

import random
import scipy.sparse as ssp
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor
from torch_geometric.utils import sort_edge_index, get_laplacian, to_scipy_sparse_matrix
from scipy import linalg


class ToSparseTensor(BaseTransform):
    def __init__(self, attr: Optional[str] = 'edge_weight', fill_cache: bool = True):
        self.attr = attr
        self.fill_cache = fill_cache

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue
            if 'original_edge_index' in store:
                edge_key = 'original_edge_index'
                sparse_size = (store.original_x.size(0), store.original_x.size(0))
            else:
                edge_key = 'edge_index'
                sparse_size = store.size()[::-1]

            nnz = store[edge_key].size(1)

            keys, values = [], []
            for key, value in store.items():
                if isinstance(value, Tensor) and value.size(0) == nnz:
                    keys.append(key)
                    values.append(value)

            store[edge_key], values = sort_edge_index(store[edge_key],
                                                      values,
                                                      sort_by_row=False)

            for key, value in zip(keys, values):
                store[key] = value

            adj_t = SparseTensor(
                row=store[edge_key][1], col=store[edge_key][0],
                value=None if self.attr is None or self.attr not in store else
                store[self.attr], sparse_sizes=sparse_size,
                is_sorted=True)

            if self.fill_cache:  # Pre-process some important attributes.
                adj_t.storage.rowptr()
                adj_t.storage.csr2csc()

            store[edge_key] = adj_t.to_symmetric()
            if self.attr is not None and self.attr in store:
                del store[self.attr]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class RandomNodeSplit(BaseTransform):
    def __init__(
            self,
            split: str = "train_rest",
            num_splits: int = 1,
            num_train_per_class: int = 20,
            num_val: Union[int, float] = 500,
            num_test: Union[int, float] = 1000,
            key: Optional[str] = "y",
    ):
        assert split in ['train_rest', 'test_rest', 'random']
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        if hasattr(store, "num_subgraphs"):
            num_nodes = store.num_subgraphs
        else:
            num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_nodes * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_nodes * self.num_test)
        else:
            num_test = self.num_test

        if self.split == 'train_rest':
            perm = torch.randperm(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val:num_val + num_test]] = True
            train_mask[perm[num_val + num_test:]] = True
        else:
            y = getattr(store, self.key)
            num_classes = int(y.max().item()) + 1
            for c in range(num_classes):
                idx = (y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                idx = idx[:self.num_train_per_class]
                train_mask[idx] = True

            remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            val_mask[remaining[:num_val]] = True

            if self.split == 'test_rest':
                test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                test_mask[remaining[num_val:num_val + num_test]] = True

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'


class RandomAddEdge(BaseTransform):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.x.size(0)

        edge_set = set(map(tuple, edge_index.transpose(0, 1).tolist()))
        num_of_new_edge = int((edge_index.size(1) // 2) * self.p)
        to_add = list()
        new_edges = random.sample(range(1, num_nodes ** 2 + 1),
                                  min(num_of_new_edge + len(edge_set) + num_nodes, num_nodes ** 2))
        c = 0
        for i in new_edges:
            if c >= num_of_new_edge:
                break
            s = ((i - 1) // num_nodes) + 1
            t = i - (s - 1) * num_nodes
            s -= 1
            t -= 1
            if s != t and (s, t) not in edge_set:
                c += 1
                to_add.append([s, t])
                to_add.append([t, s])
                edge_set.add((s, t))
                edge_set.add((t, s))
        print(f"num of added edges: {len(to_add)}")
        if len(to_add) > 0:
            new_edge_index = torch.cat([edge_index, torch.LongTensor(to_add).transpose(0, 1)], dim=1)
            data.edge_index = new_edge_index

        return data


from collections import defaultdict
from typing import Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform


class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes`)."""

    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Gather all nodes that occur in at least one edge (across all types):
        n_id_dict = defaultdict(list)
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            n_id_dict[src].append(store.edge_index[0])
            n_id_dict[dst].append(store.edge_index[1])

        n_id_dict = {k: torch.cat(v).unique() for k, v in n_id_dict.items()}

        n_map_dict = {}
        for store in data.node_stores:
            if store._key not in n_id_dict:
                n_id_dict[store._key] = torch.empty((0,), dtype=torch.long)

            idx = n_id_dict[store._key]
            mapping = idx.new_zeros(data.num_nodes)
            mapping[idx] = torch.arange(idx.numel(), device=mapping.device)
            n_map_dict[store._key] = mapping

        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            if store._key is None:
                src = dst = None
            else:
                src, _, dst = store._key

            row = n_map_dict[src][store.edge_index[0]]
            col = n_map_dict[dst][store.edge_index[1]]
            store.edge_index = torch.stack([row, col], dim=0)

        for store in data.node_stores:
            for key, value in store.items():
                if key == 'num_nodes':
                    store.num_nodes = n_id_dict[store._key].numel()

                elif store.is_node_attr(key) or key == "y":
                    store[key] = value[n_id_dict[store._key]]

        return data
