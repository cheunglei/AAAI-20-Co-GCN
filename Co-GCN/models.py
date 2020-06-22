import sys
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from torch.nn.parameter import Parameter


class Co_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3, indep_weights=False):
        super(Co_GCN, self).__init__()

        self.gc1 = GraphConvolution(input_size, hidden_size, indep_weights=indep_weights)
        self.gc2 = GraphConvolution(hidden_size, output_size, indep_weights=indep_weights)
        self.dropout = dropout
        self.m = nn.Softmax()

    def forward(self, adj, x):
        self.gc1.pi = Parameter(self.m(self.gc1.pi))
        # print(self.gc1.pi)

        x = F.relu(self.gc1(adj, x))
        # print(x)
        x = F.dropout(x, self.dropout)
        x = self.gc2(adj, x)

        return x


class MLP(nn.Module):
    def __init__(self, input_size=200, hidden_size=150, output_size=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)
        return out
