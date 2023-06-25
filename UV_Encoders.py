import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu", uv=True):
        super(UV_Encoder, self).__init__()

        self.features = features
        self.uv = uv
        max_len = max(len(x) for x in history_uv_lists.values())
        self.history_uv_lists = np.zeros((len(history_uv_lists), max_len), dtype=np.int64)
        self.history_uv_lists_mask = np.zeros((len(history_uv_lists), max_len), dtype=np.int8)
        for i, l in history_uv_lists.items():
            for j, v in enumerate(l):
                self.history_uv_lists[i, j] = v
            self.history_uv_lists_mask[i, :len(l)] = 1
        self.history_uv_lists = torch.from_numpy(self.history_uv_lists)
        self.history_uv_lists_mask = torch.from_numpy(self.history_uv_lists_mask)


        max_len = max(len(x) for x in history_r_lists.values())
        self.history_r_lists = np.zeros((len(history_r_lists), max_len), dtype=np.int64)
        self.history_r_lists_mask = np.zeros((len(history_r_lists), max_len), dtype=np.int8)
        for i, l in history_r_lists.items():
            for j, v in enumerate(l):
                self.history_r_lists[i, j] = v
            self.history_r_lists_mask[i, :len(l)] = 1
        self.history_r_lists = torch.from_numpy(self.history_r_lists)
        self.history_r_lists_mask = torch.from_numpy(self.history_r_lists_mask)

        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        tmp_history_uv = self.history_uv_lists[nodes.cpu(), :].to(self.device)
        tmp_history_uv_mask = self.history_uv_lists_mask[nodes.cpu(), :].to(self.device)
        tmp_history_r = self.history_r_lists[nodes.cpu(), :].to(self.device)
        tmp_history_r_mask = self.history_r_lists_mask[nodes.cpu(), :].to(self.device)

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_uv_mask,\
                                              tmp_history_r, tmp_history_r_mask)  # user-item network

        self_feats = self.features(nodes)
        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
