#!/usr/bin/env python
import os
import json
import pprint as pp
import torch
import wandb
from tensorboard_logger import Logger as TbLogger
from nets.gcn import Net1, Net2
from train import train
from options import get_options
from utils.data_utils import load_dataset, save_dataset, check_extension
from utils.functions import compute_embeddings


def run(opts):

    # disable sync
    os.environ['WANDB_MODE'] = 'dryrun'

    # initialize wandb
    wandb.init(project='Network1')

    # load all arguments to config to save as hyperparameters
    wandb.config.update(opts)

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(
            os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # load created dataset from path
    train_dataset = load_dataset(filename=opts.train_dataset)
    # For now, val and train dataset are same
    val_dataset = load_dataset(filename=opts.val_dataset)

    # initialize model(need to be modified for regression case)
    model = Net1(n_features=train_dataset[0].num_features, embed_dim=opts.embed_dim,
                out_features=opts.num_output).to(opts.device)

    # code for multiple gpu model(disabled for now)
    # enable once the model runs successfully for single GPU
    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     print("No. of GPUs:", torch.cuda.device_count())
    #     model = torch.nn.DataParallel(model)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # save pytorch model and track all of the gradients and optionally parameters
    wandb.watch(model, log='all')  # "gradients", "parameters", "all", or None.
    # start training
    training_status = train(model, optimizer, train_dataset, val_dataset, tb_logger, opts)
    print(training_status)

    # Get the embeddings and save to create data for Network2
    scores, embeddings = compute_embeddings(model, opts, data=train_dataset)
    # save embeddings to create dataset for Network2
    filename = os.path.join(opts.data_dir, "op{}{}_{}N{}_seed{}_label_{}_embeddings.pkl".format(
        "_{}".format(opts.data_distribution) if opts.data_distribution is not None else "",
        opts.graph_size, opts.name, opts.dataset_size, opts.dataset_seed, opts.label_type))
    
    assert opts.f or not os.path.isfile(check_extension(filename)), \
    "File already exists! Try running with -f option to overwrite."
    
    print(filename)
    save_dataset([embeddings, scores], filename)

    print("Embeddings Computed, shape:{}" .format(embeddings.shape))
    print("Scores Computed, shape:{}" .format(scores.shape))


if __name__ == '__main__':
    run(get_options())
