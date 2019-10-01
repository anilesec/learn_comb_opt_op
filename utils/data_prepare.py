import argparse
import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from utils.data_utils import check_extension, save_dataset


def compute_nodes_score(ins_sol_set):
    node_cnt = 0
    scores = []
    # M: number of solutions
    M = len(ins_sol_set['sol_set'])
    for i in range(len(ins_sol_set['loc'])):
        for sol in ins_sol_set['sol_set']:
            # a node is counted as 1 even if it is present two times in a solution.
            if i in sol:
                node_cnt += 1
        scores.append(node_cnt/M)
        node_cnt = 0

    return scores


def combine_node_features(loc, prize, depot_info):
    # concatenate loc and prize and depot info features
    nodes_loc = torch.tensor(loc)
    nodes_score = torch.unsqueeze(torch.tensor(prize), dim=1)
    nodes_feat = torch.cat((nodes_loc, nodes_score, depot_info), dim=1)

    return nodes_feat


def get_edge_index(adj, num_nodes):
    # Fully connected graph adjacency case
    import itertools
    # check if all elements are non-zero in adj
    if adj.byte().all():
        num_edges = len(list(itertools.combinations(list(range(num_nodes)), 2)))
        # multiplied by 2 as graph is undirected
        edge_index = torch.ones((2, num_edges * 2), dtype=torch.long)

    # not fully connected graph adjacecny case
    else:
        raise NotImplementedError('edge_index for not fully connected graph \
                                  adjacecny case is yet to be implemented')

    return edge_index


def create_data_object(ins, size):
    ins = generate_instance_solution(size=size,  prize_type='dist')
    depot_info = torch.zeros(graph_size+1, 1)  # added 1 for depot
    depot_info[0] = 1
    nodes_features = combine_node_features(loc=ins['loc'], prize=ins['prize'], depot_info=depot_info)
    data = Data(x=nodes_features, edge_index=get_edge_index(adj=torch.ones((20, 20)), num_nodes=20),
                y=torch.tensor(torch.unsqueeze(torch.tensor(ins['nodes_score']), dim=1)))

    # not included as it leads to batching problem(need to be looked into)
    # data.depot = ins['depot']
    # data.depot_score = ins['depot_score']
    # data.max_length = ins['max_length']
    # data.sol_set = ins['sol_set']

    return data


def make_dataset(file_name=None, size=20, num_samples=10000, offset=0, distribution_op='dist'):
    assert distribution_op is not None, "Data distribution must be specified for OP"
    # Currently the distribution can only vary in the type of the prize
    prize_type = distribution_op

    if file_name is not None:
        raise NotImplementedError('random based sol set if available for each instance case, even this need to updated')
        assert os.path.splitext(filename)[1] == '.pkl'

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            data_dict_lst = [
                {
                    'loc': torch.FloatTensor(loc),
                    'prize': torch.FloatTensor(prize),
                    'depot': torch.FloatTensor(depot),
                    'max_length': torch.tensor(max_length)
                }
                for depot, loc, prize, max_length in (data[offset:offset + num_samples])
            ]
    else:
        data_dict_lst = [
            create_data_object(ins=generate_instance_solution(size, prize_type), size=size)
            for i in tqdm(range(num_samples))
        ]

    return data_dict_lst


def generate_instance_solution(size, prize_type):
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size,))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    depot = torch.reshape(depot, shape=(1, 2))

    ins_sol_set = {
        'loc': torch.cat((depot, loc), dim=0),
        # Uniform 1 - 9, scaled by capacities
        'prize': torch.cat((torch.FloatTensor([0]), prize)),
        'depot': depot,
        'max_length': torch.tensor(MAX_LENGTHS[size])
    }
    ins_sol_set['sol_set'] = instance_sol_set(instance=ins_sol_set, n_sols=5, sol_type='random')
    ins_sol_set['nodes_score'] = compute_nodes_score(ins_sol_set)

    # score is 1 because depot is present is all the solutions
    depot_node_score = 1
    ins_sol_set['depot_score'] = depot_node_score

    return ins_sol_set


def instance_sol_set(instance, n_sols, sol_type='random'):
    if sol_type == 'random':
        ins_nodes = list(range(1, len(instance['loc'])+1))
#         sol_size = (n_sols, np.random.choice([9, 10, 11], 1, replace=False).item())
        sol_set = []
        for i in range(n_sols):
            sol_set.append(torch.as_tensor(
                np.array([0] + np.random.choice(
                    ins_nodes, np.random.choice([9, 10, 11], 1, replace=False).item(), False).tolist() + [0])))
    else:
        raise NotImplementedError("yet to implement greedy or sample based solution")

    return sol_set


def generate_op_data(dataset_size, op_size, prize_type='dist'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_instance(size, prize_type):
    # Details see paper
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size, ))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    return {
        'loc': loc,
        # Uniform 1 - 9, scaled by capacities
        'prize': prize,
        'depot': depot,
        'max_length': torch.tensor(MAX_LENGTHS[size])
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='../data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='op',
                        help="Problem, 'op' or any other type that will be implemented in future")
    parser.add_argument('--data_distribution', type=str, default='dist',
                        help="Distributions to generate for problem, default 'dist'.")

    parser.add_argument("--dataset_size", type=int, default=100000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances [20, 50, 100]")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'op': ['const', 'unif', 'dist']
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
                print("data file saved location:" .format(filename))

    # create a dataset either by reading from saved file(above) or by generating instances
    # Note: Graph of single size can be generated using make_dataset()
    # for multiple size graph dataset, call make_dataset() with diff size parameter
    print("creating dataset..")
    dataset = make_dataset(file_name=None, size=opts.graph_sizes[0],
                           num_samples=opts.dataset_size, offset=0, distribution_op=opts.data_distribution)
    print("dataset creation completed!!")

    # save dataset which contains list of pytorch geometric data objects
    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)

    if opts.filename is None:
        filename = os.path.join(datadir, "{}{}{}_{}_seed{}_data_obj_lst.pkl".format(
            problem,
            "_{}".format(opts.data_distribution) if opts.data_distribution is not None else "",
            opts.graph_sizes[0], opts.name, opts.seed))
    else:
        filename = check_extension(opts.filename)

    assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."
    print("First data object:", dataset[0])
    print(filename)
    save_dataset(dataset, filename)

