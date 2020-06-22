from __future__ import division
from __future__ import print_function

import pickle as pk
import itertools

import random
import time
import argparse
import os
import resource
import urllib.request
import json
from collections import deque
import torch
import torch.multiprocessing
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import  MyDataset
from models import GCN, MLP

'''

优点：
速度快，内存少，与原文一致，能batch

表述
受影响比较大，所以风格比较相似

记录改动用
删除 申明中的fastmode
###考虑取消dataloader的batch功能
dataset 中的label 似乎不需要那么多乱七八糟的操作 直接就可以变成tensor 
且第一个永远是0
layers 中的output = torch.spmm(adj, support) 存疑 稀疏矩阵乘法
GCN 第二层后面也加了一层dropout
cls_reg 默认0.1改成10
dataloader_workers 默认改为0


3.18 
争取实现 1. 有向图 
2. batch训练 在utils的159行附近 将动态的label数量变成固定的 其他的呢？
删去MyDataset中的 idx_train, idx_val, idx_test

3.19
andloss和orloss位置改动
论文中提出的是HE是蕴涵而不是NOT
实验证明 directed 无法训练
triplet loss 写反了 !! 写反之后竟然达到train acc 83！！  这个不知道是论文写错了or没错。 应该是论文中写错了
Semantic Regularization. 应该只针对fml F 对assignment不计算   这个应该没有疑问
还是 args.cls位置的问题 现在看好像没有放错 本质上是个调参问题

3.20 
drawLoss.py 画图需要优化 完成
quality_plot_discarded.py 导致 utils.loadData函数需要调整
quality_plot_discarded.py atoms = 3  的solution 没有生成  完成

3.21
tools/下的文件路径大多有问题，最后调整吧
draw_loss 的文件路径，最好创建个文件夹，类似与quality_plot
'''

def cuda_input(adj, features, labels):
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
    return adj, features, labels


# # to add regularization for AND OR gate
# def andloss(children):
#     temp = torch.mm(torch.t(children), children)
#     loss = torch.sum(torch.abs(temp - torch.diag(temp)))
#     # loss = torch.norm(temp - torch.diag(temp)) ** 2
#     return loss

def andloss(output,and_children):
    loss = 0
    for andgate in range(len(and_children)):
        add_child_tensor = None
        for childidx in range(len(and_children[andgate])):
            if add_child_tensor is None:
                add_child_tensor = output[and_children[andgate][childidx]]
            else:
                add_child_tensor = torch.cat(
                    (add_child_tensor, output[and_children[andgate][childidx]]))
        temp = torch.mm(torch.t(add_child_tensor), add_child_tensor)
        loss += torch.sum(torch.abs(temp - torch.diag(temp)))
        # loss += torch.norm(temp - torch.diag(temp)) ** 2
    return loss


def orloss(output,or_children):
    loss = 0
    loss = 0
    for orgate in range(len(or_children)):
        or_child_tensor = None
        for childidx in range(len(or_children[orgate])):
            if or_child_tensor is None:
                or_child_tensor = output[or_children[orgate][childidx]]
            else:
                or_child_tensor = torch.cat(
                    (or_child_tensor, output[or_children[orgate][childidx]]))
        loss += (torch.norm(torch.sum(or_child_tensor, dim=0).squeeze(0)) - 1) ** 2
        # loss += torch.norm(torch.sum(or_child_tensor, dim=0).squeeze(0) - 1) ** 2
    return loss

# def orloss(children):
#     loss = (torch.norm(torch.sum(children, dim=0).squeeze(0)) - 1) ** 2
#     # loss = torch.norm(torch.sum(children, dim=0).squeeze(0) - 1) ** 2
#     return loss



rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
torch.multiprocessing.set_sharing_strategy('file_system')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_size', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--indep_weights', action="store_true", default=False,
                    help='whether to use independent weights for different types of gates in ddnnf')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset')
parser.add_argument('--ds_path', type=str, default='../../dataset/Synthetic')
parser.add_argument('--w_reg', type=float, default=0.1, help='strength of regularization')
parser.add_argument('--cls_reg', type=float, default=0.1, help='weight of classification loss')
parser.add_argument('--directed', action='store_true', default=False)
parser.add_argument('--margin', type=float, default=1.0, help='margin in triplet margin loss')
parser.add_argument('--dataloader_workers', type=int, default=0, help='number of workers for dataloaders')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--train_rate', type=float, default=0.9, help='rate of train set')
parser.add_argument('--features', type=int, default=50, help='dim of features')
parser.add_argument('--hidden_size_mlp', type=int, default=150,
                    help='Number of hidden units of mlp.')
parser.add_argument('--embedding_size', type=int, default=100,
                    help='dim of embedding.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

for arg in vars(args):
    print(f'{arg:>30s} = {str(getattr(args, arg)):<30s}')

ds_path = args.ds_path
indep_weights = args.indep_weights
train_rate = args.train_rate
dataset = args.dataset
and_or = 'ddnnf' in dataset

# save reg in the name

reg_name = '.reg'+str(args.w_reg)
indepname = '.ind' if indep_weights else ''
directed_name = '.dir' if args.directed else ''
cls_reg_name = '.cls'+str(args.cls_reg)
seed_name = '.seed'+str(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

file_list_raw = os.listdir(f'{ds_path}/{dataset}/')
file_list = set()
for file in file_list_raw:
    if '.s' not in file and '.and' not in file and '.or' not in file and '.rel' not in file:
        file_list.add(urllib.request.unquote(file))
file_list = list(file_list)

# split train test
# shuffle first
idx = np.arange(len(file_list))
np.random.shuffle(idx)
file_list = np.array(file_list)[idx]

split_idx = int(round(len(file_list) * train_rate, 0))
file_list_train = file_list[:split_idx]
file_list_test = file_list[split_idx:]

if not os.path.exists('./testformula/'):
    os.makedirs('./testformula/')
json.dump(list(file_list_test), open('./testformula/' + dataset + seed_name + '.testformula', 'w'), ensure_ascii=False)

print('file list length: ', len(file_list), len(file_list_train), len(file_list_test))

dataset_train = MyDataset(dataset, file_list_train, and_or=and_or, args=args)
dataset_test = MyDataset(dataset, file_list_test, and_or=and_or, args=args)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=args.dataloader_workers)

print('Number of training samples:', len(dataset_train))

# Model and optimizer
model = GCN(input_size=args.features,
            hidden_size=args.hidden_size,
            output_size=args.embedding_size,
            dropout=args.dropout,
            indep_weights=indep_weights)
mlp = MLP(input_size=2*args.embedding_size,
          hidden_size=args.hidden_size_mlp,
          output_size=2,
          dropout=args.dropout)
optimizer = optim.Adam(itertools.chain(model.parameters(), mlp.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)

creterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
CE = torch.nn.CrossEntropyLoss()

loss_list = deque(maxlen=100)
loss_list_CE = deque(maxlen=100)
acc_list = deque(maxlen=100)

loss_by_iter = []

if args.cuda:
    model.cuda()
    mlp.cuda()


def train(epoch, loss_save):
    for _, data_train_group in tqdm(enumerate(dataloader_train), desc='Training', total=len(dataloader_train)):
        # pass three times first, then back propagate the loss
        model.train()
        mlp.train()
        for data_train in data_train_group:
            vector3 = []
            regularization = 0
            for data_train_item in data_train:
                # and/or children: [[1,2],[3,4]]
                adj, features, labels = cuda_input(*data_train_item[:-2])
                and_children, or_children = data_train_item[-2:]
                t = time.time()
                output = model(features.squeeze(0), adj.squeeze(0), labels.squeeze(0))
                vector3.append(output[0])

                # add regularization
                # if and_or: # and data_train_item == 0
                if and_or and data_train_item == 0:
                    # if len(and_children) != 0:
                    #         regularization += andloss(output,and_children)
                    # if len(or_children) != 0:
                    #         regularization += orloss(output,or_children)
                    regularization += andloss(output, and_children)
                    regularization += orloss(output, or_children)

            # back prop GCN
            loss_train = creterion(vector3[0].unsqueeze(0), vector3[1].unsqueeze(0), vector3[2].unsqueeze(0))
            # loss_train = creterion(vector3[0].unsqueeze(0), vector3[2].unsqueeze(0), vector3[1].unsqueeze(0))
            loss_train += args.w_reg * regularization
            loss_list.append(float(loss_train.cpu()))
            optimizer.zero_grad()
            # loss_train.backward()


            # back prop MLP
            mlp.train()
            input = torch.cat(
                (torch.cat((vector3[0], vector3[1])).unsqueeze(0), torch.cat((vector3[0], vector3[2])).unsqueeze(0)))
            pred = mlp(input)
            target = torch.LongTensor([1, 0]).cuda()
            mlp_loss = CE(pred, target)
            loss_by_iter.append(float(mlp_loss.cpu()))
            (loss_train + args.cls_reg*mlp_loss).backward()
            # (args.cls_reg * loss_train + mlp_loss).backward()
            optimizer.step()
            loss_list_CE.append(float(mlp_loss.cpu()))

            # calculate accuracy
            _, predicted = torch.max(pred.data, 1)
            acc = (predicted == target).sum().item() / target.size(0)
            acc_list.append(float(acc))

    loss_avg = np.mean(loss_list)
    CE_loss_avg = np.mean(loss_list_CE)
    acc_avg = np.mean(acc_list)

    print('Epoch: {:04d}'.format(epoch + 1),
          'Avg loss: {:.4f}'.format(loss_avg),
          'Avg CE loss: {:.4f}'.format(CE_loss_avg),
          'Avg Acc: {:.4f}'.format(acc_avg),
          'time: {:.4f}s'.format(time.time() - t))
    loss_save['triplet_loss'].append(loss_avg)
    loss_save['CE_loss'].append(CE_loss_avg)
    loss_save['acc'].append(acc_avg)

    return loss_avg, CE_loss_avg, acc_avg

def test():
    avg_loss = []
    avg_loss_CE = []
    total = 0
    correct = 0
    model.eval()
    mlp.eval()
    for _, data_test_group in tqdm(enumerate(dataloader_test), desc='Testing', total=len(dataloader_test)):
        # pass three times first, then back propagate the loss
        for data_test in data_test_group:
            vector3 = []
            for data_test_item in data_test:
                adj, features, labels = cuda_input(*data_test_item[:-2])
                output = model(features.squeeze(0), adj.squeeze(0), labels.squeeze(0))
                vector3.append(output[0])

            loss_test = creterion(vector3[0].unsqueeze(0), vector3[1].unsqueeze(0), vector3[2].unsqueeze(0))
            avg_loss.append(float(loss_test.cpu()))

            # back prop MLP
            input = torch.cat(
                (torch.cat((vector3[0], vector3[1])).unsqueeze(0), torch.cat((vector3[0], vector3[2])).unsqueeze(0)))
            pred = mlp(input)
            target = torch.LongTensor([1, 0]).cuda()
            mlp_loss = CE(pred, target)
            avg_loss_CE.append(float(mlp_loss.cpu()))

            # caclulate accuracy
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = np.average(avg_loss)
    avg_loss_CE = np.average(avg_loss_CE)
    avg_acc = correct / total
    print('Test loss: {:.4f}'.format(avg_loss),
          'Test loss CE: {:.4f}'.format(avg_loss_CE),
          'Test Acc: {:.4f}'.format(avg_acc), )
    return avg_loss, avg_loss_CE, avg_acc



# Train model
train_loss_save = {'triplet_loss': [], 'CE_loss': [], 'acc': []}
test_loss_save = {'triplet_loss': [], 'CE_loss': [], 'acc': []}
t_total = time.time()
best_acc = 0.0
for epoch in range(args.epochs):
    _ = train(epoch, train_loss_save)
    test_stats = test()
    test_loss_save['triplet_loss'].append(test_stats[0])
    test_loss_save['CE_loss'].append(test_stats[1])
    test_loss_save['acc'].append(test_stats[2])
    if test_stats[2] > best_acc:
        best_acc = test_stats[2]
        if not os.path.exists('./model_save/'):
            os.makedirs('./model_save/')
        torch.save(model, './model_save/' + dataset + reg_name+ indepname + directed_name + cls_reg_name + seed_name + '.model')
        torch.save(mlp, './model_save/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name + '.mlp.model')
        print('\tNew best model saved.')
    if not os.path.exists('./acc_loss/'):
        os.makedirs('./acc_loss/')
    json.dump(train_loss_save, open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name + '.train_save', 'w'),
              ensure_ascii=False)
    json.dump(test_loss_save, open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name + '.test_save', 'w'),
              ensure_ascii=False)
    json.dump([int(np.argmax(test_loss_save['acc'])), max(test_loss_save['acc'])],
              open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name + '.test_best', 'w'), ensure_ascii=False)

    pk.dump(loss_by_iter, open('./acc_loss/' + dataset + reg_name + indepname + directed_name + cls_reg_name + seed_name + '.loss_by_iter', 'wb'))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f"Best test acc: {max(test_loss_save['acc'])}, at epoch: {np.argmax(test_loss_save['acc'])}")
