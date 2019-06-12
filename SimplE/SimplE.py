import torch
from torch import nn
import numpy as np


class SimplE(nn.Module):
    def __init__(self):
        super(SimplE, self).__init__()

        self.n_ent = 14951
        self.n_rel = 1345
        self.K = 200

        self.device = torch.device('cuda:0')
        self.embed_eh = nn.Embedding(self.n_ent+1, self.K).to(self.device)
        self.embed_et = nn.Embedding(self.n_ent+1, self.K).to(self.device)
        self.embed_r = nn.Embedding(self.n_rel+1, self.K).to(self.device)
        self.embed_ri = nn.Embedding(self.n_rel+1, self.K).to(self.device)

        sqrt_size = 6.0 / np.sqrt(self.K)
        nn.init.uniform_(self.embed_eh.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.embed_et.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.embed_r.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.embed_ri.weight.data, -sqrt_size, sqrt_size)

    def reg_value(self):
        return ((torch.norm(self.embed_eh.weight, p=2) ** 2) + \
                (torch.norm(self.embed_et.weight, p=2) ** 2) + \
                (torch.norm(self.embed_r.weight, p=2) ** 2) + \
                (torch.norm(self.embed_ri.weight, p=2) ** 2)) / 2


    def forward(self, head, rel, tail):
        hh = self.embed_eh(head)
        ht = self.embed_et(head)
        r = self.embed_r(rel)
        ri = self.embed_ri(rel)
        th = self.embed_eh(tail)
        tt = self.embed_et(tail)

        d1 = torch.sum(hh * r * tt, dim = 1)
        d2 = torch.sum(th * ri * ht, dim = 1)
        return torch.clamp((d1 + d2) / 2, -20, 20)
