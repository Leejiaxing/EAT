import argparse
import nni
import logging
from module.model import EAT, EAT_MoE
from launcher.launcher_graph import Launcher
from module.wrapper import ModelGraph
from nni.utils import merge_parameter
from dataset.dataset import get_graph_dataset
from utils import set_global_seed

logger = logging.getLogger('sgat_brain_AutoML')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EAT_MoE", choices=["EAT_MoE", "EAT"])
    parser.add_argument("--sub_encoder", type=str, default="sage", choices=["mlp", "gcn", "gin", "sage"])
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["IMDB", ""])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--repetition", type=int, default=5)
    parser.add_argument("--loop_mode", type=str, default="cv")
    parser.add_argument("--test_ratio", type=int, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument("--wd", type=int, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=int, default=0.2)
    parser.add_argument("--node_label", type=str, default="spd")
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_sub_layers", type=int, default=5)
    parser.add_argument("--max_node_perhop", type=int, default=2)
    parser.add_argument("--lpe", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=777)

    args, _ = parser.parse_known_args()

    return args


def main(args):
    set_global_seed(args['seed'])

    dataset, meta = get_graph_dataset(
        args['dataset'], h=args['h'], node_label=args['node_label'],
        reprocess=False, max_nodes_per_hop=args['max_node_perhop']
    )
    print(args)
    print(meta)
    def model_constructor():
        if args['model'] == 'EAT':
            return EAT(
                num_features=meta['num_features'], out_channels=args["hidden_dim"], dropout=args['dropout'],
                hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'],
                use_z=True, use_rd=False, use_pe=args['lpe'], num_sub_layers=args['num_sub_layers'],
                dataset_name=args["dataset"])
        elif args['model'] == 'EAT_MoE':
            return EAT_MoE(
                num_features=meta['num_features'], out_channels=args["hidden_dim"], dropout=args['dropout'],
                hidden_dim=args["hidden_dim"], num_layers=args['num_layers'], num_heads=args['num_heads'],
                use_z=True, use_rd=False, use_pe=args['lpe'], dataset_name=args["dataset"], sub_encoder=args['sub_encoder'])
        else:
            raise NotImplementedError

    model = ModelGraph(model_constructor, model_name=args['model'], layers=args['num_layers'],
                       hiddens=args['hidden_dim'], num_classes=meta['num_classes'])
    launcher = Launcher(model=model, lr=args['lr'], wd=args['wd'],
                        dataset_name=args['dataset'])

    if args['dataset'] in ['molhiv', 'molpcba', 'moltox21', 'molbbbp', 'molbace', 'moltoxcast', 'ZINC']:
        test_mean, test_std, val_mean, val_std = launcher.train_and_test_standard_split(
            dataset, args['repetition'], args['max_epochs'], batch_size=args['batch_size'],
            tag=f"{args['dataset']} standard")

    else:
        test_mean, test_std, val_mean, val_std = launcher.train_and_test_kround(
            dataset, args['folds'], args['max_epochs'], batch_size=args['batch_size'], loop_mode=args['loop_mode'],
            test_ratio=args['test_ratio'], tag=f"{args['dataset']} {args['loop_mode']}")

    nni.report_final_result(test_mean)
    logger.debug('Final result is %g', test_mean)
    logger.debug('Send final result done.')


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
