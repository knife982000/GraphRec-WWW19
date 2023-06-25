import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F
from Functions import masked_softmax

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, to_neighs_mask):
        num_neighs = node1.shape[1] #History lenght
        uv_reps = u_rep.unsqueeze(1).repeat(1, num_neighs, 1)
        to_neighs_mask = to_neighs_mask.unsqueeze(2)
        to_neighs_mask_dim = to_neighs_mask.repeat(1, 1, self.embed_dim)
        x = torch.cat((node1, uv_reps), -1)
        x = F.relu(self.att1(x) * to_neighs_mask_dim)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x) * to_neighs_mask_dim)
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        #att = F.softmax(x * to_neighs_mask, dim=1)
        att = masked_softmax(x, to_neighs_mask).squeeze(-1)
        return att
