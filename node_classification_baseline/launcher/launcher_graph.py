import torch
import time
import numpy as np
from torch_geometric.data import DenseDataLoader, DataLoader
from utils import k_round_graph, print_time, lossAndMetric
import copy


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


class Launcher:
    def __init__(self, model, lr, wd, dataset_name):
        self.lr, self.wd = lr, wd
        self.model = model
        self.lr_decay_step_size = 100
        self.lr_decay_factor = 0.5
        self.dataset_name = dataset_name
        self.loss_metric_kit = lossAndMetric(dataset_name)

    def train_and_test_kround(self, dataset, rounds, max_epochs, batch_size, loop_mode, test_ratio=0.1, tag=""):
        val_met_folds, test_met_folds = [], []
        begin = time.time()
        metric_name = self.loss_metric_kit['metric']

        for round, (train_idx, test_idx, val_idx) in enumerate(
                zip(*k_round_graph(dataset, rounds, loop_mode, test_ratio, seed=1))):
            self.reset_launcher()
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]
            val_dataset = dataset[val_idx]
            if 'adj' in train_dataset[0]:
                train_loader = DenseDataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DenseDataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DenseDataLoader(test_dataset, batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

            best_weights = None
            min_loss = torch.inf
            max_acc = 0
            early_stop_patience = 0
            for i in range(max_epochs):
                loss_train = self.train_epoch(train_loader)
                metric_val, loss_val = self.test_epoch(val_loader)
                self.scheduler.step(loss_val)

                print(
                    '({:s} {:s}) Fold:{:d}/{:d} Epoch:{:d}({:d}) [{:s}] loss_train:{:.4f} loss_val:{:.4f} metric_val:{:.4f}'
                    .format(tag, print_time(begin), round + 1, rounds, i, 150 - early_stop_patience, metric_name,
                            loss_train, loss_val, metric_val))

                early_stop_patience += 1
                # if loss_val < min_loss:
                #     min_loss = loss_val
                #     early_stop_patience = 0
                #     best_weights = copy.deepcopy(self.model.state_dict())
                if metric_val > max_acc:
                    max_acc = metric_val
                    early_stop_patience = 0
                    best_weights = copy.deepcopy(self.model.state_dict())
                elif metric_val == max_acc:
                    if loss_val < min_loss:
                        min_loss = loss_val
                        early_stop_patience = 0
                        best_weights = copy.deepcopy(self.model.state_dict())
                if early_stop_patience == 150:
                    break

            self.model.load_state_dict(best_weights)
            metric_test, loss_test = self.test_epoch(test_loader)
            metric_val, loss_val = self.test_epoch(val_loader)

            print(
                '----------- [Test {:d}] metric_val:{:.4f} loss_val:{:.4f} metric_test:{:.4f} loss_test:{:.4f} -----------'.format(
                    round + 1, metric_val, loss_val, metric_test, loss_test)
            )
            val_met_folds.append(metric_val)
            test_met_folds.append(metric_test)

        print(
            "*********** [{:d} Fold results] {:s} metric_val:{:.2f}±{:.2f} metric_test:{:.2f}±{:.2f} ***********".format(
                rounds, metric_name,
                np.mean(val_met_folds) * 100, np.std(val_met_folds) * 100,
                np.mean(test_met_folds) * 100, np.std(test_met_folds) * 100)
        )
        return np.mean(test_met_folds), np.std(test_met_folds), np.mean(val_met_folds), np.std(val_met_folds)

    def train_and_test_standard_split(self, dataset, rounds, max_epochs, batch_size=128, tag=""):
        val_met_folds, test_met_folds = [], []
        begin = time.time()
        metric_name = self.loss_metric_kit['metric']
        for round in range(rounds):
            self.reset_launcher()
            if "mol" in self.dataset_name:
                split_idx = dataset.get_idx_split()
                train_dataset = dataset[split_idx["train"]]
                val_dataset = dataset[split_idx["valid"]]
                test_dataset = dataset[split_idx["test"]]
            else:
                train_dataset, val_dataset, test_dataset = dataset

            if 'adj' in train_dataset[0]:
                train_loader = DenseDataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DenseDataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DenseDataLoader(test_dataset, batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

            best_weights = None
            min_loss = torch.inf
            early_stop_patience = 0
            for i in range(max_epochs):
                loss_train = self.train_epoch(train_loader)
                metric_val, loss_val = self.test_epoch(val_loader)
                self.scheduler.step(loss_val)

                print(
                    '({:s} {:s}) Fold:{:d}/{:d} Epoch:{:d}({:d}) [{:s}] loss_train:{:.4f} loss_val:{:.4f} metric_val:{:.4f}'
                    .format(tag, print_time(begin), round + 1, rounds, i, 150 - early_stop_patience, metric_name,
                            loss_train, loss_val, metric_val))

                early_stop_patience += 1
                if loss_val < min_loss:
                    min_loss = loss_val
                    early_stop_patience = 0
                    best_weights = copy.deepcopy(self.model.state_dict())
                if early_stop_patience == 150:
                    break

            self.model.load_state_dict(best_weights)
            metric_test, loss_test = self.test_epoch(test_loader)
            metric_val, loss_val = self.test_epoch(val_loader)

            print(
                '----------- [Test {:d}] metric_val:{:.4f} loss_val:{:.4f} metric_test:{:.4f} loss_test:{:.4f} -----------'.format(
                    round + 1, metric_val, loss_val, metric_test, loss_test)
            )
            val_met_folds.append(metric_val)
            test_met_folds.append(metric_test)

        print(
            "*********** [{:d} Fold results] {:s} metric_val:{:.2f}±{:.2f} metric_test:{:.2f}±{:.2f} ***********".format(
                rounds, metric_name,
                np.mean(val_met_folds) * 100, np.std(val_met_folds) * 100,
                np.mean(test_met_folds) * 100, np.std(test_met_folds) * 100)
        )
        return np.mean(test_met_folds), np.std(test_met_folds), np.mean(val_met_folds), np.std(val_met_folds)

    def reset_launcher(self):
        self.model.reset_parameters()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, factor=0.2)
        self.model.to(torch.device("cuda:0"))
        torch.cuda.synchronize()

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0
        total_metric = 0
        for data in loader:
            self.optimizer.zero_grad()
            data = data.to("cuda:0")

            out = self.model(data)

            if self.dataset_name == "ZINC":
                loss_pred = metric_pred = out.view(-1)
                loss_true = metric_true = data.y.view(-1)
            elif "mol" in self.dataset_name:
                is_labeled = data.y == data.y
                loss_pred = out.float()[is_labeled]
                loss_true = data.y.float()[is_labeled]
                # metric_pred = out.float()
                # metric_true = data.y.float()
            else:
                loss_pred = out
                loss_true = data.y
                # metric_pred = out.max(1)[1].view(-1, 1)
                # metric_true = data.y.view(-1, 1)

            loss = self.loss_metric_kit['loss_fn'](loss_pred, loss_true)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # try:
            #     result = self.loss_metric_kit['evaluator'].eval({"y_true": metric_true, "y_pred": metric_pred})
            #     total_metric += result[self.loss_metric_kit['metric']] * num_graphs(data)
            # except:
            #     total_metric += 0

            # total_correct += pred.eq(data.y.view(-1)).sum().item()
        # return total_metric / len(loader.dataset), total_loss / len(loader.dataset)
        return total_loss / len(loader)

    def test_epoch(self, loader):
        self.model.eval()
        losses_true, losses_pred = [], []
        mets_true, mets_pred = [], []
        for data in loader:
            data = data.to("cuda:0")
            with torch.no_grad():
                if 'lpe' in data:
                    data.lpe_ = data.lpe

                out = self.model(data)

            if self.dataset_name == "ZINC":
                loss_pred = metric_pred = out.view(-1)
                loss_true = metric_true = data.y.view(-1)
            elif "mol" in self.dataset_name:
                is_labeled = data.y == data.y
                loss_pred = out.float()[is_labeled]
                loss_true = data.y.float()[is_labeled]
                metric_pred = out.float()
                metric_true = data.y.float()
            else:
                loss_pred = out
                loss_true = data.y.view(-1)
                metric_pred = out.max(1)[1].view(-1, 1)
                metric_true = data.y.view(-1, 1)

            losses_true.append(loss_true.detach().cpu())
            losses_pred.append(loss_pred.detach().cpu())
            mets_true.append(metric_true.detach().cpu())
            mets_pred.append(metric_pred.detach().cpu())

        losses_true = torch.cat(losses_true, dim=0)
        losses_pred = torch.cat(losses_pred, dim=0)

        mets_true = torch.cat(mets_true, dim=0).numpy()
        mets_pred = torch.cat(mets_pred, dim=0).numpy()

        loss = self.loss_metric_kit['loss_fn'](losses_pred, losses_true).item()
        met = self.loss_metric_kit['evaluator'].eval({"y_true": mets_true, "y_pred": mets_pred})[
            self.loss_metric_kit['metric']]
        return met, loss

    def test(self, dataset):
        begin = time.time()
        acc_test, loss_val = self.test_epoch(dataset, dataset.test_mask)

        print('({:s}) acc_test: {:.6f} '.format(print_time(begin), acc_test))
