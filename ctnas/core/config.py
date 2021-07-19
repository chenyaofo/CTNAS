import os
import argparse

def get_config():
    parser = argparse.ArgumentParser("CTNAS")

    parser.add_argument('--space', type=str, default="nasbench", help='The search space [nasbench|darts]')

    parser.add_argument('--output', type=str, default=None, help='output directory')
    parser.add_argument('--data', type=str, default="data/nas_bench.json", help='training data path')
    parser.add_argument('--seed', type=int, default=2020, help='seed')
    parser.add_argument('--trainset_size', type=int, default=423, help='the size of training set')
    parser.add_argument('--valset_size', type=int, default=100, help='the size of validation set')

    parser.add_argument('--train_batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=100, help='validation batch size')

    parser.add_argument('--nac_epochs', type=int, default=1000, help='num of training epochs for nac')
    parser.add_argument('--pl_iters', type=int, default=10000, help='num of training epochs for controller')

    parser.add_argument('--controller_grad_clip', type=float, default=None, help='')
    parser.add_argument('--entropy_coeff', type=float, default=5e-4, help='the coefficient of the controller')

    parser.add_argument('--n_sample_architectures', type=int, default=100, help='')

    parser.add_argument('--pseudo_ratio', type=float, default=1.0, help='')

    parser.add_argument('--n_nodes', type=int, default=5, help='the number of the nodes')
    parser.add_argument('--n_ops', type=int, default=3, help='the number of the operations')
    parser.add_argument('--n_layers', type=int, default=4, help='the number of the layers in th NAC')
    parser.add_argument('--embedding_dim', type=int, default=128, help='the dimension of embedding features')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability of dropout layer')

    parser.add_argument('--hidden_size', type=int, default=64, help='')
    parser.add_argument('--temperature', type=int, default=None, help='')
    parser.add_argument('--controller_tanh_constant', type=int, default=None, help='')
    parser.add_argument('--controller_op_tanh_reduce', type=int, default=None, help='')

    parser.add_argument('--nac_lr', type=float, default=2e-4, help='')
    parser.add_argument('--controller_lr', type=float, default=2e-4, help='')

    parser.add_argument('--n_iteration_update_pseudoset', type=int, default=100, help='')
    parser.add_argument('--evaluate_nac_freq', type=int, default=100, help='evalation frequency for nac')
    parser.add_argument('--evaluate_controller_freq', type=int, default=500, help='evalation frequency for controller')
    
    parser.add_argument('--random_baseline', action='store_true', default=False, help='randomly select baseline')

    args = parser.parse_args()

    return args

args = get_config()
if args.output is not None:
    os.makedirs(args.output, exist_ok=False)
    