from SimplE import SimplE
from dataset import Dataset
import torch
from torch import nn
import os


Epoch = 1000
lr = 0.05
Lambda = 0.1
neg_ratio = 10
kg_name = 'fb15k'


def loss(scores, labels, dataset, model):
    return torch.sum(nn.functional.softplus(-labels * scores)) + (Lambda * model.reg_value()) / dataset.batches


def train():
    dataset = Dataset(kg_name)
    model = SimplE()
    optimizer = torch.optim.Adagrad(params=model.parameters(),
                                    lr=lr)

    for iter in range(1, Epoch+1):
        tot_l = 0
        while not dataset.is_last:
            optimizer.zero_grad()
            heads, rels, tails, labels = dataset.generate_batch(1)
            scores = model.forward(heads, rels, tails)
            l = loss(scores, labels, dataset, model)
            tot_l += l.cpu().item()
            l.backward()
            optimizer.step()
        dataset.batch_idx = 0
        dataset.is_last = False
        print('Epoch #'+str(iter)+'\tloss:'+str(tot_l))
        if iter % 50 == 0:
            save_model(iter, model)



def save_model(chkpnt, model):
    print("Saving the model")
    directory = "models/fb15k/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model, directory + str(chkpnt) + ".chkpnt")

if __name__ == '__main__':
    train()