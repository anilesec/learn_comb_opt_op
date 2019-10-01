import math
import torch
import pickle
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils.data_utils import load_pkl_file, save_dataset, check_extension


def create_dataset(fname_ins_sols_dataset, fname_ins_embeddings):
    ins_dataset = load_pkl_file(fname_ins_sols_dataset)
    ins_dataset_score_embedds = load_pkl_file(fname_ins_embeddings)
    # ins_dataset_scores = instances_dataset_score_embedds[0]
    # ins_dataset_embeddings = instances_dataset_score_embedds[1]
    
    dataset = []
    for ins, ins_scores, ins_embeds in tqdm(enumerate(ins_dataset, ins_dataset_scores, ins_dataset_embeddings)):
        print(ins)
        print(ins_scores)
        print(ins_embeds)
        break
        dataset.append(zip(ins, ins_scores, ins_embeds))

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_filename",
                        help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--fname_ins_sol_dataset",
                        help="Filename of datset of instances and their M solutions")
    parser.add_argument("--fname_ins_embeddings",
                        help="Filename of embeddings of instsances in --fname_ins_sol_dataset, \
                        scores also present in this file along with embeddings")
    parser.add_argument("--data_dir", default='data/op',
                        help="Create datasets in data_dir/op (default 'data')")
    parser.add_argument("--name", type=str, required=True,
                        help="Name to identify dataset")
    parser.add_argument('--data_distribution', type=str, default='dist',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("-f", action='store_true',
                        help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1111,
                        help="Random seed, 1111:train, 2222:val, 3333:test")

    parser.add_argument("--dataset_size", type=int, default=10000,
                        help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances (default 20, 50, 100)")

    opts = parser.parse_args()

    assert opts.dataset_filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
                "Can only specify filename when generating a single dataset"

    if opts.dataset_filename is None:
        filename = os.path.join(opts.data_dir, "op_k_sols{}{}_{}_seed{}.pkl".format(
                "_{}".format(opts.data_distribution) if opts.data_distribution is not None else "",
                opts.graph_sizes[0], opts.name, opts.seed))
    else:
        filename = check_extension(opts.dataset_filename)

    assert opts.f or not os.path.isfile(check_extension(filename)), \
    "File already exists! Try running with -f option to overwrite."

    print(filename)
    save_dataset(dataset, filename)
    print("Network2 dataset creation completed!!")
    print("Saved location:" .format(filename))

