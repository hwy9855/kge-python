import numpy as np
import pickle as pk
from random import sample
from copy import deepcopy

data_path = '../data/fb15k/'

Lambda = 0.01
Ita = 1
K = 50
Epoch = 15000
batch_size = 150
n_ent = 14951
n_rel = 1345
# d = L1
# Optimal configuration for FB15K in paper


def l1norm(vec):
    '''
    :param vec: a vector
    :return: l1-norm of vec
    '''

    return np.linalg.norm(vec, 1)


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
        triple = [head, relation, tail]
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
    '''
    :param n: batch size
    :param N: batch range
    :return: a mini-batch
    '''

    # minibatch = []
    # for i in range(n):
    #     k = int(np.random.rand() * N)
    #     while k in minibatch:
    #         k = int(np.random.rand() * N)
    #     minibatch.append(k)
    return sample(range(N), n)

def run(train_set, valid_set):
    embed_ent = np.zeros((n_ent, K))
    embed_rel = np.zeros((n_rel, K))

    for i in range(n_ent):
        embed_ent[i] = np.random.uniform(-6/np.sqrt(K), 6/np.sqrt(K), K)
        embed_ent[i] = embed_ent[i] / np.linalg.norm(embed_ent[i])
    for i in range(n_rel):
        embed_rel[i] = np.random.uniform(-6/np.sqrt(K), 6/np.sqrt(K), K)
        embed_rel[i] = embed_rel[i] / np.linalg.norm(embed_rel[i])
    # init

    n_train = len(train_set)

    for i in range(Epoch):
        if i % 100 == 0:
            print(i)
        minibatch = batch(batch_size, n_train)

        T_batch = []
        for S in minibatch:
            pair = []
            pair.append(train_set[S])
            S_broke = deepcopy(train_set[S])
            if np.random.randint(0, 2) == 0:
                S_broke[0] = np.random.randint(0, n_ent - 1)
            else:
                S_broke[2] = np.random.randint(0, n_ent - 1)
            pair.append(S_broke)
            T_batch.append(pair)

        for pair in T_batch:
            head = int(pair[0][0])
            rel = int(pair[0][1])
            tail = int(pair[0][2])
            headCurrpted = int(pair[1][0])
            tailCurrpted = int(pair[1][2])
            tmpPos = embed_ent[head] + embed_rel[rel] - embed_ent[tail]
            tmpNeg = embed_ent[headCurrpted] + embed_rel[rel] - embed_ent[tailCurrpted]
            for i in range(K):
                if tmpPos[i] >= 0:
                    tmpPos[i] = 1
                else:
                    tmpPos[i] = -1
                if tmpNeg[i] >= 0:
                    tmpNeg[i] = 1
                else:
                    tmpNeg[i] = -1
            tmpPos = - 2 * Lambda * tmpPos
            tmpNeg = - 2 * Lambda * tmpNeg

            embed_ent[head] = embed_ent[head] + tmpPos
            embed_rel[rel] = embed_rel[rel] + tmpPos - tmpNeg
            embed_ent[tail] = embed_ent[tail] - tmpPos
            embed_ent[headCurrpted] = embed_ent[headCurrpted] - tmpNeg
            embed_ent[tailCurrpted] = embed_ent[tailCurrpted] + tmpNeg

            embed_ent[head] = embed_ent[head] / np.linalg.norm(embed_ent[head])
            embed_rel[rel] = embed_rel[rel] / np.linalg.norm(embed_rel[rel])
            embed_ent[tail] = embed_ent[tail] / np.linalg.norm(embed_ent[tail])
            embed_ent[headCurrpted] = embed_ent[headCurrpted] / np.linalg.norm(embed_ent[headCurrpted])
            embed_ent[tailCurrpted] = embed_ent[tailCurrpted] / np.linalg.norm(embed_ent[tailCurrpted])

    return embed_ent, embed_rel


def model2pk(embed_ent, embed_rel):
    with open('./TransE/embedded_entity.pickle', 'wb') as f_ent:
        pk.dump(embed_ent, f_ent)
    with open('./TransE/embedded_rel.pickle', 'wb') as f_rel:
        pk.dump(embed_rel, f_rel)


if __name__ == '__main__':
    train_set, valid_set = readData()
    embed_ent, embed_rel = run(train_set, valid_set)
    model2pk(embed_ent, embed_rel)
