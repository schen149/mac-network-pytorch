import sys
import pickle
from collections import Counter

import os
import json

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_with_constraints import CLEVR_w_constraints, collate_data, transform, collate_data_validation
from constrained_model import MACNetwork

from constraints import constraint_loss_fn_calc

batch_size = 64
n_epoch = 8
dim = 512

# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch,
          constraints_alpha=0.02):
    clevr = CLEVR_w_constraints(sys.argv[1], transform=transform)
    train_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    tick = 0
    net.train(True)
    running_acc = []
    for image, question, q_len, answer, _, c_mask in pbar:
        image, question, answer, c_mask = (
            image.to(device),
            question.to(device),
            answer.to(device),
            c_mask.to(device),
        )

        net.zero_grad()
        output, mid_layer_attn = net(image, question, q_len)
        loss = criterion(output, answer)
        constraint_loss = constraint_loss_fn_calc(mid_layer_attn, c_mask)
        # constraint_loss = constraint_loss.detach()
        loss = loss + constraints_alpha * constraint_loss / batch_size
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch + 1, loss.item(), moving_loss
            )
        )

        tick += 1
        if tick % 20 == 0:
            running_acc.append(tick)

        accumulate(net_running, net)

    clevr.close()
    return running_acc

def valid(epoch):
    clevr = CLEVR_w_constraints(sys.argv[1], 'val', transform=None)
    valid_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data_validation
    )
    dataset = iter(valid_set)

    net_running.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output, _ = net_running(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        for k, v in family_total.items():
            w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    print(
        'Avg Acc: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values())
        )
    )

    clevr.close()


if __name__ == '__main__':


    if len(sys.argv) != 2:
        print("Usage: python ... [] []", file=sys.stderr)
        exit(1)

    train_path = sys.argv[1]

    data_dir = "../data/keywords_only"
    with open(os.path.join(data_dir, 'dic.pkl'), 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, dim).to(device)
    net_running = MACNetwork(n_words, dim).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(n_epoch):
        running_acc = train(epoch)
        print(json.dumps(running_acc))

        valid(epoch)



        with open(
            '../checkpoint/checkpoint_{}_orig.model'.format(str(epoch + 1).zfill(2)), 'wb'
        ) as f:
            torch.save(net_running.state_dict(), f)
