import os
import nni
import argparse
import logging
from nni.utils import merge_parameter
from launcher.launcher_node import Launcher
from module.wrapper import ModelNode
from module.model import EAT_MoE, EAT
from dataset.dataset import get_node_dataset
from utils import set_global_seed

logger = logging.getLogger('sgat_brain_AutoML')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EAT_MoE", choices=["EAT_MoE", "EAT"])
    parser.add_argument("--sub_encoder", type=str, default="mlp", choices=["mlp", "gcn", "gin", "sage"])
    parser.add_argument("--dataset", type=str, default="Computers", choices=["cora", "Computers", "CS"])
    parser.add_argument("--tag", type=str, default="normal")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loop_mode", type=str, default="random")
    parser.add_argument("--test_ratio", type=float, default=0.6)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--node_label", type=str, default="spd")
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_sub_layers", type=int, default=2)
    parser.add_argument("--max_node_perhop", type=int, default=3)
    parser.add_argument("--lpe", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--edge_p", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")

    args, _ = parser.parse_known_args()

    return args


def main(args: dict):
    set_global_seed(args['seed'])

    datasets, meta = get_node_dataset(
        args["dataset"], h=args["h"], node_label=args["node_label"],
        use_rd=True, reprocess=False, max_nodes_per_hop=args["max_node_perhop"],
        test_ratio=args["test_ratio"], edge_p=args['edge_p']
    )
    print(args)
    print(meta)

    def model_constructor():
        if args['model'] == 'EAT':
            return EAT(
                num_features=meta['num_features'], out_channels=meta["num_classes"], dropout=args['dropout'],
                hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'],
                use_z=True, use_rd=True, use_pe=args['lpe'], num_sublayers=args['num_sub_layers'], dataset_name=args["dataset"])
        elif args['model'] == 'EAT_MoE':
            return EAT_MoE(
                num_features=meta['num_features'], out_channels=meta["num_classes"], dropout=args['dropout'],
                hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'],
                use_z=True, use_rd=True, use_pe=args['lpe'], dataset_name=args["dataset"], sub_encoder=args['sub_encoder'])
        else:
            raise NotImplementedError

    model = ModelNode(model_constructor, args['model'])

    launcher = Launcher(
        model=model, lr=args['lr'], wd=args['wd'],
        dataset_name=args["dataset"], model_name=args['model'], device=args['device'])

    test_mean, test_std, val_mean, val_std = launcher.train_and_test_standard_split(
        datasets, args['folds'], args['max_epochs'],
        tag=f"{args['model']} {args['tag']} {args['dataset']}")

    nni.report_final_result(test_mean)
    logger.debug('Final result is %g', test_mean)
    logger.debug('Send final result done.')


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
