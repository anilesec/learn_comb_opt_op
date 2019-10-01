import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser()

    # Model and Training
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Node embedding dimension.')
    parser.add_argument('--num_output', type=int, default=1,
                        help='Number of output units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--output_dir', default='outputs',
                        help='Directory to write output models to')
    parser.add_argument('--log_dir', default='logs',
                        help='Directory to write TensorBoard information to')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='The number of epochs to train')
    parser.add_argument('--log_step', type=int, default=1,
                        help='Log info every log_step steps')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable logging TensorBoard files')
    parser.add_argument('--run_name', default='run',
                        help='Name to identify the run')

    # Data
    parser.add_argument('--dataset_seed', type=int, required=True,
                        help="Random seed, 1111:train, 2222:val, 3333:test")  
    parser.add_argument("--dataset_size", type=int, required=True,
                        help="Size of the dataset")
    parser.add_argument("--name", type=str, required=True,
                        help="Name to identify dataset")
    parser.add_argument('--data_distribution', type=str, default='dist',
                        help="Distributions to generate for problem, 'dist, unif, const'.")
    parser.add_argument("--data_dir", default='data/op',
                        help="Create datasets in data_dir/op (default 'data')")
    parser.add_argument('--train_dataset', required=True,
                        help='training dataset file path')
    parser.add_argument('--problem', default='op',
                        help="The problem to solve, default 'op'")
    parser.add_argument('--graph_size', type=int, required=True,
                        help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of instances per batch during training')
    parser.add_argument('--val_dataset', type=str, default=True,
                        help='Dataset file to use for validation')
    parser.add_argument("-f", action='store_true',
                        help="Set true to overwrite")
    parser.add_argument("--label_type", type=str,
                        required=True, help="type of labels 'binary' or 'score'")

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )

    return opts
