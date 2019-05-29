import numpy as np
import pickle as pk
from random import sample
from copy import deepcopy

data_path = '../data/fb15k/'

Lambda = 0.01
Ita = 1
K = 50
Epoch = 500
batch_size = 150
batches = 100
n_ent = 14951
n_rel = 1345

headrel = {}
tailrel = {}
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

def run(train_set):
    embed_ent = np.zeros((n_ent, K))
    embed_rel = np.zeros((n_rel, K))

    for i in range(n_ent):
        embed_ent[i] = np.random.uniform(-6/np.sqrt(K), 6/np.sqrt(K), K)
        embed_ent[i] = embed_ent[i] / np.linalg.norm(embed_ent[i])
    for i in range(n_rel):
        embed_rel[i] = np.random.uniform(-6/np.sqrt(K), 6/np.sqrt(K), K)
        embed_rel[i] = embed_rel[i] / np.linalg.norm(embed_rel[i])
        # do not normalize the relation embedding

    # init

    n_train = len(train_set)

    for i in range(Epoch):
        # if i % 10 == 0:
        res = 0
        for k in range(batches):
            minibatch = batch(int(n_train/batches), n_train)

            # minibatch = np.arange(n_train)
            # np.random.shuffle(minibatch)
            T_batch = []
            # for S in minibatch:


            # loss

            for k in range(len(minibatch)):
                S = minibatch[k]
                pair = []
                pair.append(train_set[S])
                S_broke = deepcopy(train_set[S])
                broke = np.random.randint(0, n_ent - 1)
                if np.random.randint(0, 2) == 0:
                    while broke in tailrel[(S_broke[2], S_broke[1])]:
                        broke = np.random.randint(0, n_ent - 1)
                    S_broke[0] = broke
                else:
                    while broke in headrel[(S_broke[0], S_broke[1])]:
                        broke = np.random.randint(0, n_ent - 1)
                    S_broke[2] = broke
                pair.append(S_broke)
                T_batch.append(pair)

            for pair in T_batch:
                head = int(pair[0][0])
                rel = int(pair[0][1])
                tail = int(pair[0][2])
                headCurrpted = int(pair[1][0])
                tailCurrpted = int(pair[1][2])
                tmpPos = embed_ent[head] + embed_rel[rel] - embed_ent[tail]
                tmpPosRes = l1norm(tmpPos)
                tmpNeg = embed_ent[headCurrpted] + embed_rel[rel] - embed_ent[tailCurrpted]
                tmpNegRes = l1norm(tmpNeg)
                if tmpPosRes + Ita > tmpNegRes:
                    res += tmpPosRes + Ita - tmpNegRes
                    for j in range(K):
                        if tmpPos[j] >= 0:
                            tmpPos[j] = 1
                        else:
                            tmpPos[j] = -1
                        if tmpNeg[j] >= 0:
                            tmpNeg[j] = 1
                        else:
                            tmpNeg[j] = -1
                    tmpPos = - Lambda * tmpPos
                    tmpNeg = - Lambda * tmpNeg

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
        if i % 20 == 19:
            model2pk(embed_ent, embed_rel)
        print('epoch #' + str(i) + '\t loss:' + str(res))
    return embed_ent, embed_rel


def model2pk(embed_ent, embed_rel):
    with open('./TransE/embedded_entity.pickle', 'wb') as f_ent:
        pk.dump(embed_ent, f_ent)
    with open('./TransE/embedded_rel.pickle', 'wb') as f_rel:
        pk.dump(embed_rel, f_rel)


if __name__ == '__main__':
    train_set, valid_set = readData()
    embed_ent, embed_rel = run(train_set)
    model2pk(embed_ent, embed_rel)
