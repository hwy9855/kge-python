import numpy as np
import torch

class Dataset():
    def __init__(self, kg):
        super(Dataset, self).__init__()
        self.data_path = '../data/'
        self.train_path = self.data_path + kg + '/train_processed.txt'
        self.test_path = self.data_path + kg + '/test_processed.txt'
        self.valid_path = self.data_path + kg + '/valid_processed.txt'
        self.read()
        self.batch_idx = 0
        self.batches = 100
        self.n = self.train_set.shape[0]
        self.batch_size = int(self.n / self.batches) + 1
        self.ent_size = 14951
        self.is_last = False
        self.device = 'cuda:0'
        self.num_ent = 14591
        self.num_rel = 1345

    def read(self):
        with open(self.train_path) as tf:
            lines = tf.readlines()

            self.train_set = np.zeros((len(lines), 3))
            for i, line in enumerate(lines):
                self.train_set[i][0] = int(line.split('\t')[0])
                self.train_set[i][1] = int(line.split('\t')[1])
                self.train_set[i][2] = int(line.split('\t')[2])

        with open(self.test_path) as test_file:
            lines = test_file.readlines()

            self.test_set = np.zeros((len(lines), 3))
            for i, line in enumerate(lines):
                self.test_set[i][0] = int(line.split('\t')[0])
                self.test_set[i][1] = int(line.split('\t')[1])
                self.test_set[i][2] = int(line.split('\t')[2])

        with open(self.valid_path) as vf:
            lines = vf.readlines()

            self.valid_set = np.zeros((len(lines), 3))
            for i, line in enumerate(lines):
                self.valid_set[i][0] = int(line.split('\t')[0])
                self.valid_set[i][1] = int(line.split('\t')[1])
                self.valid_set[i][2] = int(line.split('\t')[2])


    def generate_neg(self, pos):
        neg = np.random.random_integers(0, self.ent_size, 1)
        while neg == pos:
            neg = np.random.random_integers(0, self.ent_size, 1)
        return neg

    def generate_neg_batch(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=1)
        for i in range(neg_batch.shape[0]):
            if np.random.normal(0, 1) > 0:
                neg_batch[i][0] = self.generate_neg(neg_batch[i][0])
            else:
                neg_batch[i][2] = self.generate_neg(neg_batch[i][2])
            neg_batch[i][-1] = -1
        return neg_batch

    def generate_pos_batch(self):
        if self.batch_idx + self.batch_size > self.n:
            pos_batch = self.train_set[self.batch_idx:]
            self.is_last = True
        else:
            pos_batch = self.train_set[self.batch_idx:self.batch_idx + self.batch_size]
            self.batch_idx += self.batch_size
        return np.append(pos_batch, np.ones((len(pos_batch), 1)), axis=1).astype("int")

    def generate_batch(self, neg_ratio):
        pos_batch = self.generate_pos_batch()
        neg_batch = self.generate_neg_batch(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads = torch.Tensor(batch[:, 0]).long().to(self.device)
        rels = torch.Tensor(batch[:, 1]).long().to(self.device)
        tails = torch.Tensor(batch[:, 2]).long().to(self.device)
        labels = torch.Tensor(batch[:, 3]).float().to(self.device)
        return heads, rels, tails, labels

if __name__ == '__main__':
    dataset = Dataset('fb15k')
    print(dataset.generate_batch(1))