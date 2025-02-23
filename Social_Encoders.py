import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from Functions import masked_independent_shuffle, masked_shuffle

class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.max_len = torch.from_numpy(np.asarray([len(x) for x in social_adj_lists.values()])).long()
        max_len = torch.max(self.max_len).item()
        self.social_adj_lists = np.zeros((len(social_adj_lists), max_len), dtype=np.int64)
        self.social_mask = np.zeros((len(social_adj_lists), max_len), dtype=np.int8)
        for i, l in social_adj_lists.items():
            for j, v in enumerate(l):
                self.social_adj_lists[i, j] = v
            self.social_mask[i, :len(l)] = 1
        self.social_adj_lists = torch.from_numpy(self.social_adj_lists)
        self.social_mask = torch.from_numpy(self.social_mask)
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes: torch.Tensor):
        cpu_nodes = nodes.cpu()
        max_size = max(torch.max(self.max_len[cpu_nodes]).item(), 2)
        to_neighs = self.social_adj_lists[cpu_nodes, :max_size].to(self.device)
        to_neighs_mask = self.social_mask[cpu_nodes, :max_size].to(self.device)

        #Reduce neighbors
        if max_size > 64:
            to_neighs = masked_independent_shuffle(to_neighs, to_neighs_mask)[:, :64]
            to_neighs_mask = to_neighs_mask[:, :64]

        neigh_feats = self.aggregator.forward(nodes, to_neighs, to_neighs_mask)  # user-user network

        self_feats = self.features(nodes).to(self.device)
        self_feats = self_feats.t()
        
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
