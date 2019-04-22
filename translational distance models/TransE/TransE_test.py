import numpy as np
import pickle as pk

data_path = '../data/fb15k/'
Hits = 10
n_ent = 14951


def l1norm(head, relation, tail):
    return -np.linalg.norm(head + relation - tail, 1)

def readData():
    test_set = []

    test = open(data_path + 'train_processed.txt')
    lines = test.readlines()
    for line in lines:
        head = line.split('\t')[0]
        relation = line.split('\t')[1]
        tail = line.split('\t')[2].split('\n')[0]
        triple = [head, relation, tail]
        test_set.append(triple)

    return test_set


def loadModel():
    with open('./TransE/embedded_entity.pickle', 'rb') as f_ent:
        embed_ent = pk.load(f_ent)
    with open('./TransE/embedded_rel.pickle', 'rb') as f_rel:
        embed_rel = pk.load(f_rel)
    return embed_ent, embed_rel


def runHits(test_set, embed_ent, embed_rel):
    iter = 0
    test_size = len(test_set)
    hits_head = 0
    hits_tail = 0
    for triple in test_set:
        head = embed_ent[int(triple[0])]
        relation = embed_rel[int(triple[1])]
        tail = embed_ent[int(triple[2])]
        d_headCorrupted = []
        d_tailCorrupted = []
        for i in range(n_ent):
            d_headCorrupted.append(l1norm(embed_ent[i], relation, tail))
            d_tailCorrupted.append(l1norm(head, relation, embed_ent[i]))
        d_headCorrupted.sort(reverse=True)
        d_tailCorrupted.sort(reverse=True)
        if l1norm(head, relation, tail) >= d_headCorrupted[Hits]:
            hits_head += 1
        if l1norm(head, relation, tail) >= d_tailCorrupted[Hits]:
            hits_tail += 1
        iter += 1
        if iter % 10 == 0:
            print(str(iter) + ': ' + str(hits_head) + ' ' + str(hits_tail))
    print('Hits@' + str(Hits) + ': ')
    print('head\t' + str(hits_head/test_size))
    print('tail\t' + str(hits_tail/test_size))

def runMean(test_set, embed_ent, embed_rel):
    iter = 0
    test_size = len(test_set)
    hits_head = 0
    hits_tail = 0
    for triple in test_set:
        head = embed_ent[int(triple[0])]
        relation = embed_rel[int(triple[1])]
        tail = embed_ent[int(triple[2])]
        d_headCorrupted = []
        d_tailCorrupted = []
        for i in range(n_ent):
            d_headCorrupted.append(l1norm(embed_ent[i], relation, tail))
            d_tailCorrupted.append(l1norm(head, relation, embed_ent[i]))
        d_headCorrupted.sort(reverse=True)
        d_tailCorrupted.sort(reverse=True)
        for i in range(n_ent):
            if l1norm(head, relation, tail) >= d_headCorrupted[i]:
                print("left:\t" + str(i))
                break
        for i in range(n_ent):
            if l1norm(head, relation, tail) >= d_tailCorrupted[i]:
                print("right:\t" + str(i))
                break

if __name__ == '__main__':
    test_set = readData()
    embed_ent, embed_rel = loadModel()
    runHits(test_set, embed_ent, embed_rel)

