import os
import argparse
from launcher.launcher_graph import Launcher
from graph_baseline.gcn_graph import GCN
from graph_baseline.gat_graph import GAT
from graph_baseline.sage_graph import SAGE
from graph_baseline.gin_graph import GIN
from graph_baseline.gatv2_graph import GATV2
from graph_baseline.super_gat_graph import SuperGAT
from dataset.dataset import get_dataset
from utils import set_global_seed


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gin")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--repetition", type=int, default=5)
    parser.add_argument("--loop_mode", type=str, default="cv")
    parser.add_argument("--test_ratio", type=int, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--wd", type=int, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=int, default=0.2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=888)

    args, _ = parser.parse_known_args()

    return args


def main(args: dict):
    set_global_seed(args['seed'])

    datasets, meta = get_dataset(args["dataset"])
    print(args)
    print(meta)

    if args['model'] == 'gcn':
        model = GCN(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    elif args['model'] == 'gat':
        model = GAT(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    elif args['model'] == 'gatv2':
        model = GATV2(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    elif args['model'] == 'super':
        model = SuperGAT(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    elif args['model'] == 'sage':
        model = SAGE(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    elif args['model'] == 'gin':
        model = GIN(
            num_features=meta['num_features'], num_classes=meta['num_classes'], dropout=args['dropout'],
            hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], dataset_name=args['dataset'])
    else:
        raise NotImplementedError

    launcher = Launcher(model=model, lr=args['lr'], wd=args['wd'], dataset_name=args["dataset"])

    if args['dataset'] in ['molhiv', 'molpcba', 'moltox21', 'molbbbp', 'molbace', 'moltoxcast', 'ZINC']:
        test_mean, test_std, val_mean, val_std = launcher.train_and_test_standard_split(
            datasets, args['repetition'], args['max_epochs'], batch_size=args['batch_size'],
            tag=f"{args['dataset']} standard")

    else:
        test_mean, test_std, val_mean, val_std = launcher.train_and_test_kround(
            datasets, args['folds'], args['max_epochs'], batch_size=args['batch_size'], loop_mode=args['loop_mode'],
            test_ratio=args['test_ratio'], tag=f"{args['dataset']} {args['loop_mode']}")


if __name__ == "__main__":
    params = vars(get_params())
    print(params)
    main(params)

