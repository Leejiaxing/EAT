import torch
import random
import torch.nn.functional as F
import scipy.sparse as ssp
from scipy.linalg import eigh
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import StratifiedKFold
import time
from torch._utils import _accumulate
import ogb
import ogb.lsc
import ogb.graphproppred
import ogb.nodeproppred


def lossAndMetric(dataset_name):
    if dataset_name == 'molpcba':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'ap',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molpcba'),
        }
    elif dataset_name == 'molhiv':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molhiv'),
        }
    elif dataset_name == 'moltox21':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-moltox21'),
        }
    elif dataset_name == 'molbbbp':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molbbbp'),
        }
    elif dataset_name == 'molbace':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-molbace'),
        }
    elif dataset_name == 'moltoxcast':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-moltoxcast'),
        }
    elif dataset_name == 'ZINC':
        return {
            'loss_fn': F.l1_loss,
            'metric': 'mae',
            'metric_mode': 'min',
            'evaluator': ogb.lsc.PCQM4MEvaluator()
        }
    elif dataset_name == 'arxiv':
        return {
            'loss_fn': F.cross_entropy,
            'metric': 'acc',
            'metric_mode': 'max',
            'evaluator': ogb.nodeproppred.Evaluator("ogbn-arxiv")
        }
    elif dataset_name == 'products':
        return {
            'loss_fn': F.cross_entropy,
            'metric': 'acc',
            'metric_mode': 'max',
            'evaluator': ogb.nodeproppred.Evaluator("ogbn-products")
        }
    elif dataset_name == 'proteins':
        return {
            'loss_fn': F.binary_cross_entropy_with_logits,
            'metric': 'rocauc',
            'metric_mode': 'max',
            'evaluator': ogb.nodeproppred.Evaluator("ogbn-proteins")
        }
    else:
        # 节点分类、图分类
        return {
            'loss_fn': F.cross_entropy,
            'metric': 'acc',
            'metric_mode': 'max',
            'evaluator': ogb.graphproppred.Evaluator('ogbg-ppa')
        }


def save_result(result_file, trials):
    print("正在保存结果...", result_file)
    with open(result_file, "w+") as f:
        for result in trials.results:
            if 'loss' in result and result['loss'] <= trials.best_trial['result']['loss']:
                print(result, file=f)
    print("结果已保存 {:s}".format(result_file))
    print(trials.best_trial)


def k_round_node(dataset, rounds, test_ratio=None):
    if test_ratio is None:
        # 交叉验证
        skf = StratifiedKFold(rounds, shuffle=True)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(dataset.data.y)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx))

        val_indices = [test_indices[i - 1] for i in range(rounds)]

        for i in range(rounds):
            train_mask = torch.ones(len(dataset.data.y), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            train_indices.append(train_mask.nonzero().view(-1))
        return train_indices, val_indices, test_indices
    else:
        # 随机划分
        split_indice = []
        for i in range(rounds):
            split_ratio = [1 - 2 * test_ratio, test_ratio, test_ratio]
            num_training = int(len(dataset.data.y) * split_ratio[0])
            num_test = int(len(dataset.data.y) * split_ratio[1])
            num_val = len(dataset.data.y) - (num_training + num_test)

            lengths = [num_training, num_val, num_test]
            indices = torch.randperm(
                sum(lengths), generator=torch.Generator().manual_seed(i)).tolist()
            split_indice.append([indices[offset - length: offset]
                                 for offset, length in zip(_accumulate(lengths), lengths)])
        return split_indice  # train val test


def k_round_graph(dataset, rounds, loop_mode, test_ratio, seed):
    if loop_mode == "cv":
        # 交叉验证
        skf = StratifiedKFold(rounds, shuffle=True, random_state=seed)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
            test_indices.append(torch.from_numpy(idx).long())

        val_indices = [test_indices[i - 1] for i in range(rounds)]

        for i in range(rounds):
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))

        return train_indices, val_indices, test_indices
    else:
        # 随机划分
        split_indice = [[], [], []]
        for i in range(rounds):
            split_ratio = [1 - 2 * test_ratio, test_ratio, test_ratio]
            num_training = int(len(dataset) * split_ratio[0])
            num_test = int(len(dataset) * split_ratio[1])
            num_val = len(dataset) - (num_training + num_test)

            lengths = [num_training, num_val, num_test]
            indices = torch.randperm(sum(lengths), generator=torch.Generator().manual_seed(i)).tolist()
            for i, (offset, length) in enumerate(zip(_accumulate(lengths), lengths)):
                split_indice[i].append(indices[offset - length: offset])
            # [indices[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)]
        return split_indice  # train val test


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        _, out_nei, _ = ssp.find(A[node, :])
        in_nei, _, _ = ssp.find(A[:, node])
        nei = set(out_nei).union(set(in_nei))
        res = res.union(nei)
    return res


class return_prob(object):
    def __init__(self, steps=50):
        self.steps = steps

    def __call__(self, data):
        adj = to_scipy_sparse_matrix(
            data.edge_index, num_nodes=data.num_nodes).tocsr()
        adj += ssp.identity(data.num_nodes, dtype='int', format='csr')
        rp = np.empty([data.num_nodes, self.steps])
        inv_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
        inv_deg.setdiag(1 / adj.sum(1))
        P = inv_deg * adj
        if self.steps < 5:
            Pi = P
            for i in range(self.steps):
                rp[:, i] = Pi.diagonal()
                Pi = Pi * P
        else:
            inv_sqrt_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
            inv_sqrt_deg.setdiag(1 / (np.array(adj.sum(1)) ** 0.5))
            B = inv_sqrt_deg * adj * inv_sqrt_deg
            L, U = eigh(B.todense())
            W = U * U
            Li = L
            for i in range(self.steps):
                rp[:, i] = W.dot(Li)
                Li = Li * L

        data.rp = torch.FloatTensor(rp)

        return data


def print_time(begin):
    fin_time = time.time()
    current_time = time.strftime("%m/%d %H:%M", time.localtime(fin_time))
    duration = time.strftime("%dd %H:%M:%S", time.gmtime(fin_time - begin))
    return "[{:s} +{:s}]".format(current_time, duration)


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
