from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import random
import time
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from utils import  MyDataset
from models import Co_GCN

model = Co_GCN(input_size=4, hidden_size=10, output_size=2)
epoches = 500
LR = 0.001

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

file_path = 'dataset'
# dataset_split = [0,1,2,3,4]
dataset_split = [0,1,2,3,4]
test_dataset_split = [5,6]
dataset_train = MyDataset(file_path, dataset_split)
dataset_test = MyDataset(file_path, test_dataset_split)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)


loss_each = []
correct_each = []
for i in range(epoches):
    loss_avg = 0.0
    for index, batch in enumerate(dataloader_train):
        model.train()
        # print(batch)
        adjs, features, y = batch
        adjs = torch.squeeze(adjs,0)
        features = torch.squeeze(features, 0)

        prediction = model(adjs, features)
        y = y.long()
        loss = loss_func(prediction[0].unsqueeze(0),y)
        # print(prediction[0].unsqueeze(0),y)
        loss_avg += loss.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct_sum = 0.0

    for index, batch in enumerate(dataloader_train):
        model.eval()
        adjs, features, y = batch
        adjs = torch.squeeze(adjs,0)
        features = torch.squeeze(features, 0)

        prediction = model(adjs, features)
        y = y.long()
        prediction = prediction[0].unsqueeze(0)
        # print(torch.argmax(prediction,dim=1),y)
        correct = float(torch.sum(torch.argmax(prediction,dim=1)==y)*1.0/len(y))
        # print(prediction[0].unsqueeze(0),y)
        correct_sum += correct


    loss_avg /= len(dataloader_train)
    # print(loss_avg)
    loss_each.append(loss_avg)
    correct_sum /= len(dataloader_train)
    # print(correct_sum)
    correct_each.append(correct_sum)

print(correct_each)
plt.plot(loss_each,label='loss')
plt.plot(correct_each,label='correct')
plt.savefig('./1k.png')
plt.show()
