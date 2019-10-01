import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Net1(nn.Module):
    """
    ****For large graphs, replace GCN with GraphSage Network,
     as per the paper it has resulted in good results.****
    """
    def __init__(self, n_features, embed_dim=128, out_features=1):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(n_features, embed_dim, cached=False)
        self.conv2 = GCNConv(embed_dim, embed_dim, cached=False)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=out_features)

    # # regression setting
    # def forward(self, data, compute_embeddings=False):
    #     x = F.relu(self.conv1(data.x, data.edge_index))
    #     x = F.dropout(x, training=self.training)
    #     embeddings = F.relu(self.conv2(x, data.edge_index))  # relu operation is not present in original paper implementation

    #     if not compute_embeddings:
    #         out = F.tanh(self.fc1(embeddings))

    #     return out, embeddings

    # classification setting
    def forward(self, data, compute_embeddings=False):
        x = F.tanh(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)
        embeddings = torch.tanh(self.conv2(x, data.edge_index))  # relu operation is not present in original paper implementation

        if not compute_embeddings:
            out = torch.sigmoid(self.fc1(embeddings))

        return out, embeddings


class Net2(nn.Module):
    """
    ****For large graphs, replace GCN with GraphSage Network,
     as per the paper it has resulted in good results.****
    """
    def __init__(self, n_features, embed_dim=128, out_features=1): 
        super(Net2, self).__init__()
        self.conv1 = ChebConv(n_features, embed_dim, K=2, cached=False)
        self.conv2 = ChebConv(embed_dim, embed_dim, K=2, cached=False)

    # # regression setting
    # def forward(self, data, compute_embeddings=False):
    #     x = F.relu(self.conv1(data.x, data.edge_index))
    #     x = F.dropout(x, training=self.training)
    #     embeddings = F.relu(self.conv2(x, data.edge_index))  # relu operation is not present in original paper implementation

    #     if not compute_embeddings:
    #         out = F.tanh(self.fc1(embeddings))

    #     return out, embeddings

    # classification setting
    def forward(self, data, compute_embeddings=False):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)
        embeddings = self.conv2(x, data.edge_index)  # relu operation is not present in original paper implementation

        if not compute_embeddings:
            out = torch.sigmoid(self.fc1(embeddings))

        return out, embeddings