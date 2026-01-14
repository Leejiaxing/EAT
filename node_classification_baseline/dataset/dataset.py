import sys
import os
import torch
import time
from os import path
from shutil import rmtree

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CitationFull, WebKB, WikipediaNetwork, ZINC, Coauthor, Amazon, Actor, TUDataset, Planetoid
import torch_geometric.transforms as T
from transform import RandomNodeSplit, ToSparseTensor,RandomAddEdge, RemoveIsolatedNodes


sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/' % os.path.dirname(os.path.realpath(__file__)))


def get_dataset(name, reprocess=False, clean=False, edge_p=0, **args):
    root = "data"
    pre_transform = [T.ToUndirected(), RemoveIsolatedNodes()]
    # pre_transform = []
    dir_name = "origin"

    if edge_p > 0:
        dir_name = f"p{edge_p}_" + dir_name
        edge_trans = RandomAddEdge(edge_p)
        pre_transform.insert(0, edge_trans)

    pre_transform = T.Compose(pre_transform)
    root = path.join(root, dir_name)
    preprocessed_dir = os.path.join(root, name, "processed")
    if reprocess and path.isdir(preprocessed_dir):
        print("removing preprocessing dir:", preprocessed_dir)
        rmtree(preprocessed_dir)

    print(root)
    citation = ['dblp']
    planetoid = ['cora',  'citeseer', 'pubmed']
    webkb = ["Cornell", "Texas"]
    amazon = ['Computers', 'photo']
    wiki = ["chameleon", "squirrel", "crocodile"]
    ogbn = ['arxiv', 'products', 'proteins']
    coauthor = ["CS", "Physics"]
    film = ['film']
    zinc = ["ZINC"]
    tu_dataset = ["PROTEINS", "MUTAG", "DD", "ENZYMES", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI", "PTC_MR"]
    ogbg = ['molhiv', 'molpcba', 'moltox21', 'molbbbp',  'molbace', 'moltoxcast']

    num_features = None
    num_classes = None

    btime = time.time()
    if name in tu_dataset:
        datasets = [TUDataset(root, name)]
    elif name in citation:
        datasets = [CitationFull(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                              pre_transform=pre_transform)]
    elif name in planetoid:
        datasets = [Planetoid(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                                 pre_transform=pre_transform)]
    elif name in film:
        datasets = [Actor(root, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                                 pre_transform=pre_transform)]
    elif name in webkb:
        datasets = [WebKB(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                          pre_transform=pre_transform)]
    elif name in amazon:
        datasets = [Amazon(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']), pre_transform=pre_transform)]
    elif name in wiki:
        datasets = [WikipediaNetwork(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                                     pre_transform=pre_transform, geom_gcn_preprocess=False)]
    elif name in coauthor:
        datasets = [Coauthor(root, name, transform=T.RandomNodeSplit(num_val=0.2, num_test=args['test_ratio']),
                             pre_transform=pre_transform)]
    elif name in zinc:
        datasets = [
            ZINC(subset=True, root=os.path.join(root, "zinc"), split="train", pre_transform=pre_transform),
            ZINC(subset=True, root=os.path.join(root, "zinc"), split="val", pre_transform=pre_transform),
            ZINC(subset=True, root=os.path.join(root, "zinc"), split="test", pre_transform=pre_transform)
        ]
        num_features = 64
        num_classes = 1
    elif name in ogbg:
        datasets = [PygGraphPropPredDataset(f'ogbg-{name}', root=root, pre_transform=pre_transform)]
        num_classes = datasets[0].num_tasks
    elif name in ogbn:
        datasets = [
            PygNodePropPredDataset(f'ogbn-{name}', root=root, transform=ToSparseTensor(), pre_transform=pre_transform)]
    else:
        raise RuntimeError("数据集名称错误")
    meta = {
        'num_features': num_features if num_features is not None else datasets[0].num_features,
        'num_classes': num_classes if num_classes is not None else datasets[0].num_classes,
        'num_graphs': 0,
        'num_nodes': 0,
        'num_subnodes': 0,
        'mean_subnodes': 0,
        'mean_degree': 0,
        'preprocessing_time': time.time() - btime
    }
    for dataset in datasets:
        meta['num_graphs'] += len(dataset)
        meta["mean_degree"] += dataset.data.edge_index.size(1)
        meta['num_nodes'] += dataset.data.num_nodes
    meta["mean_degree"] = meta["mean_degree"] / meta['num_nodes']
    if len(datasets) == 1:
        datasets = datasets[0]
    return datasets, meta
