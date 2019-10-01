import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader


def compute_embeddings(model, opts, data):
    """
    Function to compute the embeddings after training
    :param model: trained model
    :param opts: arg parse arguments
    :param data: data to be used
    :return: node scores and embeddings
    """
    node_embeddings = []
    node_scores = []
    # batch size is 1 for computing embeddings
    dataloader = DataLoader(dataset=data, batch_size=1, shuffle=True, num_workers=16)
    model.eval()
    print("computing embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch.to(opts.device)
            batch_scores, batch_embeddings = model(batch, compute_embeddings=False)
            node_embeddings.append(batch_embeddings)
            node_scores.append(batch_scores)
            # input('enter for embeddings')
            # print(node_embeddings)

    return torch.stack(node_scores), torch.stack(node_embeddings)
#     return torch.stack(node_scores).reshape(opts.dataset_size, opts.graph_size+1, -1),
#     torch.stack(node_embeddings).reshape(opts.dataset_size, opts.graph_size+1, -1)


def accuracy(output, target, threshold=0.5):
    """
    Computes the accuracy for multiple binary predictions
    output: predicted probabilities
    target: labels(could be probabilities as well)
    threshold: classification threshold
    """
    pred = (output >= threshold).type(torch.cuda.FloatTensor)
    accu = torch.sum(pred == target).type(torch.cuda.FloatTensor)

    return accu / target.numel()
