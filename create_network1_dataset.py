import torch
import os
import argparse
import itertools
from tqdm import tqdm
from utils.data_utils import load_pkl_file, save_dataset, check_extension
from torch_geometric.data import Data


def compute_nodes_score(ins_sol_set):
    """
    Function to create nodes score using top k sol set
    :param ins_sol_set: instance with solution set along with node scores
    :return: nodes score for single instance
    """
    node_cnt = 0
    scores = []
    # M: number of solutions
    M = len(ins_sol_set)
    for i in range(len(ins_sol_set['loc'])):
        for sol in ins_sol_set['sol_set']:
            # a node is counted as 1 even if it is present two times in a solution.
            if i in sol:
                node_cnt += 1
        scores.append(node_cnt/M)
        node_cnt = 0

    return scores


def compute_nodes_label(ins_sol_set):
    """
    function to create binary node labels based on approx sol
    :param ins_sol_set: instance with solution set along with node scores
    :return: nodes label for single instance
    """
    nodes_label = torch.zeros(size=(len(ins_sol_set['loc']), 1))
    # input('enter')
    # print(nodes_label)
    # input('enter')
    # only one(first) of k sols are used to create labels
    nodes_label[ins_sol_set['sol_set'][0]] = 1.0
    # print(nodes_label)
    # input('enter')
    nodes_label_numpy = nodes_label.numpy()
    # print(nodes_label_numpy)
    # input( 'enter' )
    # print(list( itertools.chain( *nodes_label_numpy ) ))
    # input( 'enter' )

    return list(itertools.chain(*nodes_label_numpy))


def get_edge_index(adj, num_nodes):
    """
    Function to compute edge index
    :param adj: adjacency matrix of graph instance
    :param num_nodes: total number of nodes of graph instance
    :return: edge_index
    """
    # Fully connected graph adjacency case
    import itertools
    # check if all elements are non-zero in adj
    if adj.byte().all():
        num_edges = len(list(itertools.combinations(list(range(num_nodes)), 2)))
        # multiplied by 2 as graph is undirected
        edge_index = torch.ones((2, num_edges * 2), dtype=torch.long)

    # not fully connected graph adjacency case
    else:
        raise NotImplementedError('edge_index for not fully connected graph \
                                  adjacency case is yet to be implemented')

    return edge_index


def combine_node_features(loc, prize, depot_vec):
    """
    Function to combine different node features
    :param loc: location fature
    :param prize: prize feature
    :param depot_vec: depot info feature
    :return: a matrix of combined features
    """
    # concatenate loc and prize and depot info features
    depot_vector = torch.unsqueeze(depot_vec, dim=1)
    nodes_prize = torch.unsqueeze(prize, dim=1)
    nodes_feat = torch.cat((loc, nodes_prize, depot_vector), dim=1)

    return nodes_feat


def create_data_object(label_type, ins_sol_set):
    """
    Function to create pytorch geometric data object
    :param label_type: type of true label; binary or score
    :param ins_sol_set: instance with k sols
    :return: pytorch geometric data object
    """
    depot_info = torch.zeros(len(ins_sol_set['loc']))  # added +1 for depot
    depot_info[0] = 1
    nodes_features = combine_node_features(loc=ins_sol_set['loc'], prize=ins_sol_set['prize'], depot_vec=depot_info)
    if label_type == 'score':
        data_obj = Data(x=nodes_features, edge_index=get_edge_index(adj=torch.ones((20, 20)),
                        num_nodes=len(ins_sol_set['loc'])),
                        y=torch.tensor(torch.unsqueeze(torch.tensor(ins_sol_set['nodes_label_score']), dim=1)))
    elif label_type == 'binary':
        data_obj = Data(x=nodes_features, edge_index=get_edge_index(adj=torch.ones((20, 20)),
                        num_nodes=len(ins_sol_set['loc'])),
                        y=torch.tensor(torch.unsqueeze(torch.tensor(ins_sol_set['nodes_label_binary']), dim=1)))
    else:
        raise NotImplementedError("Unknown label_type, specify label_type 'score' or 'binary'")

    # not included as it leads to batching problem(need to be looked into)
    # data.depot = ins_sol_set['depot']
    # data.max_length = ins_sol_set['max_length']
    # data.sol_set = ins_sol_set['sol_set']

    return data_obj


def combine_instance_solution(ins, sol_set):
    """
    Function to combine single instance and top k solutions
    and computed node scores
    :param ins: single graph instance
    :param sol_set: top k sols of single graph instance
    :return: instance with solution set along with node scores
    """
    depot = torch.tensor(ins[0], dtype=torch.float).reshape(shape=(1,2))
    loc = torch.tensor(ins[1], dtype=torch.float)
    prize = torch.tensor(ins[2], dtype=torch.float)
    max_length = torch.tensor(ins[3], dtype=torch.float)
    # noinspection PyDictCreation
    ins_sol = {
        'loc': torch.cat((depot, loc), dim=0),  # concatenated depot location
        'prize': torch.cat((torch.FloatTensor([0]), prize)),  # concatenated depot prize(0)
        'depot': depot,
        'max_length': max_length
    }
    ins_sol['sol_set'] = sol_set  # instance_sol_set(instance=ins_sol_set, n_sols=5, sol_type='AM')
    ins_sol['nodes_label_score'] = compute_nodes_score(ins_sol)
    # nodes label is just based on one approx solution for one instance
    ins_sol['nodes_label_binary'] = compute_nodes_label(ins_sol)

    # input('enter for scores')
    # print(ins_sol['nodes_label_score'])
    # input('enter for binary labels')
    # print(ins_sol['nodes_label_binary'])
    # input("press to continue")

    return ins_sol


def create_dataset(label_type, instances_filename, approx_sols_filename):
    """
    Function to create dataset using generated instance file and
    corresponding approximate instance sols file
    :param label_type:
    :param instances_filename: generated instance filename
    :param approx_sols_filename: approximate instance sols filename
    :return: dataset
    """
    instances = load_pkl_file(instances_filename)
    instances_topk_sols = load_pkl_file(approx_sols_filename)
    assert len(instances) == len(instances_topk_sols), "miss-match in number of instances and sols, should be same"
    dataset = []
    for i in tqdm(range(len(instances))):
        ins_sol_set = combine_instance_solution(ins=instances[i], sol_set=instances_topk_sols[i])
        dataset.append(create_data_object(label_type, ins_sol_set))
        
    return dataset


# code to create dataset for network1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename",
                        help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data/op',
                        help="Create datasets in data_dir/op (default 'data')")
    parser.add_argument("--ins_fname", default='data/op/op_dist20_trainN10000_seed1111.pkl',
                        help="generated instances filename")
    parser.add_argument("--approx_sols_fname", default='data/op/op_dist20_trainN10000_seed1111_top_k_sol_set.pkl',
                        help="generated instances filename")
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
    parser.add_argument("--label_type", type=str, required=True,
                        help="type of labels 'binary' or 'score'")

    opts = parser.parse_args()

    assert opts.ins_fname is not None or opts.approx_sols_fname is not None, \
        "specify the filename for instances and approximate solutions"

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    if opts.filename is None:
        filename = os.path.join(opts.data_dir, "op_k_sols{}{}_{}N{}_seed{}_label_{}.pkl".format(
            "_{}".format(opts.data_distribution) if opts.data_distribution is not None else "",
            opts.graph_sizes[0], opts.name, opts.dataset_size, opts.seed, opts.label_type))
    else:
        filename = check_extension(opts.filename)

    print(filename)
    assert opts.f or not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

    data_set = create_dataset(opts.label_type,  opts.ins_fname, opts.approx_sols_fname)
    print("Dataset creation completed!!")
    print("Storing dataset...")
    save_dataset(data_set, filename)
    print("Dataset stored")
    print('Stored location:{}' .format(filename))
    print("Number of data objects: {}" .format(len(data_set)))
