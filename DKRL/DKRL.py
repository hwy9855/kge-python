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

        # cuda switch
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # load word_vec
        self.word_vec = word_vec
        self.n_w = 100

        # init params for CNN, using optimal parameters in the paper
        self.k_1 = 2
        self.k_2 = 1
        self.pooling = 4
        self.input_size = 343  # longest description has 343 words
        self.n_1 = 100

        # init relation embedding
        # entity embedding will not be stored
        self.embed_r = nn.Embedding(self.n_rel+1, self.n).to(self.device)
        sqrt_size = 6.0 / np.sqrt(self.n)
        nn.init.uniform_(self.embed_r.weight.data, -sqrt_size, sqrt_size)

        # init CNN
        self.W_1 = torch.zeros(self.n_1, self.k_1 * self.n_w).to(self.device)
        self.W_2 = torch.zeros(self.n, self.k_2 * self.n_1).to(self.device)
        nn.init.xavier_normal_(self.W_1)
        nn.init.xavier_normal_(self.W_2)

    def rel_embed(self, rel):
        return self.embed_r[rel]

    def forward(self, ent_words):
        '''
        CNN forward function
        :param ent_words: words descriptions of entity
        :return: embedding output of CNN
        '''

        pooling_length = int((len(ent_words) - 1) / self.pooling) + 1

        # init embed_vec
        embed_vec_1 = torch.zeros(pooling_length, self.n_1)
        embed_vec_2 = torch.zeros(self.n)

        # first CNN layer & pooling layer

        for i in range(pooling_length):
            for n_fil in range(self.n_1):
                max_pool = 0
                for j in range(self.pooling):
                    if i * self.pooling + j < len(ent_words):
                        word_embed = torch.from_numpy(self.word_vec[ent_words[i * self.pooling + j]])
                        word_embed = word_embed.float()

                        tmp = 0
                        for window in range(self.k_1):
                            curr_idx = window * self.n_w
                            tmp += sum(word_embed * self.W_1[n_fil][curr_idx:curr_idx+self.n_w])

                        if tmp > max_pool:
                            max_pool = tmp
                embed_vec_1[i][n_fil] = max_pool

        # second CNN layer & pooling layer

        for i in range(pooling_length):
            for n_fil in range(self.n):
                for window in range(self.k_2):
                    curr_idx = i * self.n_1
                    tmp_embed = sum(embed_vec_1[i] * self.W_2[n_fil][curr_idx:curr_idx+self.n_1])
                embed_vec_2[n_fil] = tmp_embed
        embed_vec_2 /= pooling_length

        return embed_vec_2
