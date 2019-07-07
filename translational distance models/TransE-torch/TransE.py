import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from random import sample
from copy import deepcopy
import datetime

data_path = '../data/fb15k/'

Lambda = 0.01
Epoch = 500
batch_size = 150
batches = 100
headrel = {}
tailrel = {}

torch_num = []
for i in range(14951):
    torch_num.append(torch.Tensor([i]).long().cuda())

class TransE(nn.Module):
    def __init__(self):

        super(TransE, self).__init__()
        self.device = torch.device('cuda:0')
        self.margin = torch.Tensor([1]).to(self.device)
        self.K = 50
        self.n_ent = 14951
        self.n_rel = 1345

        self.embed_e = nn.Embedding(self.n_ent, self.K).to(self.device)
        self.embed_r = nn.Embedding(self.n_rel, self.K).to(self.device)

        sqrt_size = 6.0 / np.sqrt(self.K)
        nn.init.uniform_(self.embed_e.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.embed_r.weight.data, -sqrt_size, sqrt_size)


    def forward(self, heads, rels, tails):
        embed_heads = self.embed_e(heads)
        embed_rels = self.embed_r(rels)
        embed_tails = self.embed_e(tails)

        scores = torch.norm(embed_heads + embed_rels - embed_tails, p = 1)
        return scores

    def loss(self, pos, neg):
        return -min(0, pos - neg + self.margin)

def readData():
    train_set = []
    valid_set = []
    train = open(data_path + 'train_processed.txt')
    valid = open(data_path + 'valid_processed.txt')
    lines = train.readlines()
    for line in lines:
        head = line.split('\t')[0]
        relation = line.split('\t')[1]
        tail = line.split('\t')[2].split('\n')[0]
        triple = [torch_num[int(head)], torch_num[int(relation)], torch_num[int(tail)]]

        if (head, relation) not in headrel:
            headrel[(head, relation)] = []
        headrel[(head, relation)].append(int(tail))
        if (tail, relation) not in tailrel:
            tailrel[(tail, relation)] = []
        tailrel[(tail, relation)].append(int(head))

        train_set.append(triple)

    lines = valid.readlines()
    for line in lines:
        head = line.split('\t')[0]
        relation = line.split('\t')[1]
        tail = line.split('\t')[2].split('\n')[0]
        triple = [head, relation, tail]
        valid_set.append(triple)
    return train_set, valid_set

def batch(n, N):
    return sample(range(N), n)

def run(train_set):
    n_train = len(train_set)
    n_ent = 14951
    n_rel = 1345

    model = TransE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for iter in range(Epoch):

        totl = 0
        for batchs in range(batches):
            minibatch = batch(int(n_train / batches), n_train)
            T_batch = []
            for k in range(len(minibatch)):
                S = minibatch[k]
                pair = []
                pair.append(train_set[S])
                S_broke = deepcopy(train_set[S])
                broke = np.random.randint(0, n_ent - 1)
                if np.random.randint(0, 2) == 0:
                    # while broke in tailrel[(S_broke[2], S_broke[1])]:
                    #     broke = np.random.randint(0, n_ent - 1)
                    S_broke[0] = torch_num[broke]
                else:
                    # while broke in headrel[(S_broke[0], S_broke[1])]:
                    #     broke = np.random.randint(0, n_ent - 1)
                    S_broke[2] = torch_num[broke]
                pair.append(S_broke)
                T_batch.append(pair)

            l = torch.Tensor([0]).cuda()
            for pair in T_batch:
                hp = pair[0][0]
                rp = pair[0][1]
                tp = pair[0][2]
                hn = pair[1][0]
                rn = pair[1][1]
                tn = pair[1][2]
                optimizer.zero_grad()
                out1 = model.forward(hp, rp, tp)
                out2 = model.forward(hn, rn, tn)
                l += model.loss(out1, out2)
            totl += l
            l.backward(torch.ones_like(l))
            optimizer.step()
            print(batchs, totl)
        print('Epoch #'+str(iter)+'\t'+'Loss = '+str(totl))

if __name__ == '__main__':
    train_set, valid_set = readData()
    embed_ent, embed_rel = run(train_set)
    # model2pk(embed_ent, embed_rel)