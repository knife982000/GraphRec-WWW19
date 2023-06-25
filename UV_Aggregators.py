import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_uv_mask, history_r, history_r_mask):
        dim_mask = history_uv_mask.unsqueeze(2).repeat(1, 1, self.embed_dim)
        if self.uv == True:
            e_uv = self.v2e(history_uv) * dim_mask
            uv_rep = self.u2e(nodes)
        else: 
            e_uv = self.u2e(history_uv) * dim_mask
            uv_rep = self.v2e(nodes)
        e_r = self.r2e(history_r) * dim_mask
        x = torch.cat((e_uv, e_r), -1)
        x = F.relu(self.w_r1(x))
        o_history = F.relu(self.w_r2(x))

        att_w = self.att(o_history, uv_rep, history_uv_mask)
        att_history = torch.sum(o_history * att_w.unsqueeze(-1), dim=1)
        return att_history
