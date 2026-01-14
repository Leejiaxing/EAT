import os
import argparse
from launcher.launcher_node import Launcher
from node_baseline.gcn import GCN
from node_baseline.gat import GAT
from node_baseline.gatv2 import GATv2
from node_baseline.sage import Sage
from node_baseline.gin import GIN
from node_baseline.sgc import SGC
from node_baseline.ssg import SSG
from node_baseline.super_gat import SuperGAT
from dataset.dataset import get_dataset
from utils import set_global_seed


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="super")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--tag", type=str, default="normal")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loop_mode", type=str, default="random")
    parser.add_argument("--test_ratio", type=float, default=0.6)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--edge_p", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    return args


def main(args: dict):
    set_global_seed(args['seed'])

    datasets, meta = get_dataset(args["dataset"], edge_p=args['edge_p'], test_ratio=args["test_ratio"])
    print(args)
    print(meta)

    if args['model'] == 'gcn':
        model = GCN(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'])
    elif args['model'] == 'gat':
        model = GAT(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'])
    elif args['model'] == 'gatv2':
        model = GATv2(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'])
    elif args['model'] == 'super':
        model = SuperGAT(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'])
    elif args['model'] == 'sage':
        model = Sage(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'])
    elif args['model'] == 'gin':
        model = GIN(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'])
    elif args['model'] == 'sgc':
        model = SGC(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'])
    elif args['model'] == 'ssg':
        model = SSG(
            in_channel=meta['num_features'], out_channel=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'])
    else:
        raise NotImplementedError

    launcher = Launcher(
        model=model, lr=args['lr'], wd=args['wd'],
        dataset_name=args["dataset"], save_logits=os.path.join("logits", args["model"], args["dataset"])
    )

    test_mean, test_std, val_mean, val_std = launcher.train_and_test_standard_split(
        datasets, args['folds'], args['max_epochs'],
        tag=f"{args['model']} {args['tag']} {args['dataset']}")


if __name__ == "__main__":
    params = vars(get_params())
    print(params)
    main(params)

