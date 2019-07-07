import numpy as np
import torch
from torch import nn

class DKRL(nn.Module):
    def __init__(self, word_vec):
        super(DKRL, self).__init__()

        # init params for krl
        self.n_ent = 14951
        self.n_rel = 1345
        self.n = 100

        # cuda on
        self.device = torch.device('cuda:0')

        # load word_vec
        self.word_vec = word_vec
        self.n_w = 100

        # init params for CNN, using optimal parameters in the paper
        self.k_1 = 2
        self.k_2 = 1
        self.input_size = 343  # longest description has 343 words
        self.n_1 = 100

        # init relation embedding
        # entity embedding will not be stored
        self.embed_r = nn.Embedding(self.n_rel+1, self.n).to(self.device)
        sqrt_size = 6.0 / np.sqrt(self.n)
        nn.init.uniform_(self.embed_r.weight.data, -sqrt_size, sqrt_size)

        # init CNN
        self.W_1 = torch.Tensor(self.n_1, self.k_1 * self.n_w).to(self.device)
        self.W_2 = torch.Tensor(self.n, self.k_2 * self.n_1).to(self.device)
        nn.init.xavier_normal_(self.W_1)
        nn.init.xavier_normal_(self.W_2)


    def forward(self, *input):
        pass