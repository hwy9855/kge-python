# import pandas as pd
import numpy as np

dataset_path = '../data/fb15k/FB15k/'

if __name__ == '__main__':
    train = open(dataset_path + 'freebase_mtr100_mte100-train.txt')
    test = open(dataset_path + 'freebase_mtr100_mte100-test.txt')
    valid = open(dataset_path + 'freebase_mtr100_mte100-valid.txt')

    lines = train.readlines()
    entities = dict()
    relations = dict()
    idx_ent = 0
    idx_rel = 0
    for line in lines:
        head = line.split('\t')[0]
        relation = line.split('\t')[1]
        tail = line.split('\t')[2].split('\n')[0]
        if head not in entities:
            entities[head] = idx_ent
            idx_ent += 1
        if tail not in entities:
            entities[tail] = idx_ent
            idx_ent += 1
        if relation not in relations:
            relations[relation] = idx_rel
            idx_rel += 1
    with open('../data/fb15k/entities2id.txt', 'w') as f:
        for entity in entities:
            f.write(entity + '\t' + str(entities[entity]) + '\n')
    with open('../data/fb15k/relations2id.txt', 'w') as f:
        for relation in relations:
            f.write(relation + '\t' + str(relations[relation]) + '\n')


    train = open(dataset_path + 'freebase_mtr100_mte100-train.txt')
    test = open(dataset_path + 'freebase_mtr100_mte100-test.txt')
    valid = open(dataset_path + 'freebase_mtr100_mte100-valid.txt')
    with open('../data/fb15k/train_processed.txt', 'w') as f:
        lines = train.readlines()
        for line in lines:
            f.write(str(entities[line.split('\t')[0]]))
            f.write('\t')
            f.write(str(relations[line.split('\t')[1]]))
            f.write('\t')
            f.write(str(entities[line.split('\t')[2].split('\n')[0]]))
            f.write('\n')
    with open('../data/fb15k/test_processed.txt', 'w') as f:
        lines = test.readlines()
        for line in lines:
            f.write(str(entities[line.split('\t')[0]]))
            f.write('\t')
            f.write(str(relations[line.split('\t')[1]]))
            f.write('\t')
            f.write(str(entities[line.split('\t')[2].split('\n')[0]]))
            f.write('\n')
    with open('../data/fb15k/valid_processed.txt', 'w') as f:
        lines = valid.readlines()
        for line in lines:
            f.write(str(entities[line.split('\t')[0]]))
            f.write('\t')
            f.write(str(relations[line.split('\t')[1]]))
            f.write('\t')
            f.write(str(entities[line.split('\t')[2].split('\n')[0]]))
            f.write('\n')

