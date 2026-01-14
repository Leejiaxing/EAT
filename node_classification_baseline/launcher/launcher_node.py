import torch
import time
import numpy as np
from utils import print_time, lossAndMetric
import copy


class Launcher:
    def __init__(self, model, lr, wd, dataset_name, save_logits=None):
        self.lr, self.wd = lr, wd
        self.model = model
        self.lr_decay_step_size = 100
        self.lr_decay_factor = 0.5
        self.dataset_name = dataset_name
        self.loss_metric_kit = lossAndMetric(dataset_name)
        self._save_dir = save_logits

    def train_and_test_standard_split(self, dataset, rounds, max_epochs, tag=""):
        val_met_folds, test_met_folds = [], []
        begin = time.time()
        metric_name = self.loss_metric_kit['metric']
        data = dataset[0]
        data = data.to("cuda:0")

        if "ogbn" in dataset.name:
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to("cuda:0")
            valid_idx = split_idx['valid'].to("cuda:0")
            test_idx = split_idx['test'].to("cuda:0")
        else:
            train_idx = data.train_mask.to("cuda:0")
            valid_idx = data.val_mask.to("cuda:0")
            test_idx = data.test_mask.to("cuda:0")

        train_times = []
        test_times = []
        for round in range(rounds):
            self.reset_launcher()
            best_weights = None
            min_loss = torch.inf
            early_stop_patience = 0
            for i in range(max_epochs):
                btime = time.time()
                loss_train, out_train = self.train_epoch(data, train_idx)
                train_times.append(time.time() - btime)

                metric_val, loss_val = self.test_epoch(data, valid_idx)
                self.scheduler.step(loss_val)

                print('({:s} {:s}) Fold:{:d}/{:d} Epoch:{:d}({:d}) [{:s}] loss_train:{:.4f} loss_val:{:.4f} '
                      'metric_val:{:.4f}'
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

            btime = time.time()
            metric_test, loss_test = self.test_epoch(data, test_idx)
            test_times.append(time.time() - btime)

            metric_val, loss_val = self.test_epoch(data, valid_idx)

            print('----------- [Test {:d}] metric_val:{:.4f} loss_val:{:.4f} metric_test:{:.4f} loss_test:{:.4f} '
                  '-----------'.format(
                round + 1, metric_val, loss_val, metric_test, loss_test)
            )
            val_met_folds.append(metric_val)
            test_met_folds.append(metric_test)

        print("*********** [{:d} Fold results] {:s} metric_val:{:.2f}±{:.2f} metric_test:{:.2f}±{:.2f} "
              "training_time:{:.2f}±{:.2f} testing_time:{:.2f}±{:.2f} ***********".format(
            rounds, metric_name,
            np.mean(val_met_folds) * 100, np.std(val_met_folds) * 100,
            np.mean(test_met_folds) * 100, np.std(test_met_folds) * 100,
            np.mean(train_times) * 100, np.std(train_times) * 100,
            np.mean(test_times) * 100, np.std(test_times) * 100
        ))
        return np.mean(test_met_folds), np.std(test_met_folds), np.mean(val_met_folds), np.std(val_met_folds)

    def print_model_size(self):
        print("The number of trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        size_model = 0
        for param in self.model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        print(f"Model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

    def reset_launcher(self):
        self.model.reset_parameters()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100, factor=0.2)
        self.model.to(torch.device("cuda:0"))
        self.print_model_size()
        torch.cuda.synchronize()

    def train_epoch(self, data, mask):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data)
        loss_pred = out[mask]
        loss_true = data.y[mask].view(-1)

        loss = self.loss_metric_kit['loss_fn'](loss_pred, loss_true)

        loss.backward()
        self.optimizer.step()

        return loss.item(), out

    def test_epoch(self, data, mask):
        self.model.eval()

        with torch.no_grad():
            out = self.model(data)

        loss_pred = out[mask]
        loss_true = data.y[mask].view(-1)

        metric_pred = out.argmax(dim=-1, keepdim=True)[mask]
        metric_true = data.y[mask].view(-1, 1)

        loss = self.loss_metric_kit['loss_fn'](loss_pred, loss_true)
        met = self.loss_metric_kit['evaluator'].eval({"y_true": metric_true, "y_pred": metric_pred})[
            self.loss_metric_kit['metric']]

        return met, loss.item()

    def test(self, dataset):
        begin = time.time()
        acc_test, loss_val = self.test_epoch(dataset, dataset.test_mask)

        print('({:s}) acc_test: {:.6f} '.format(print_time(begin), acc_test))
