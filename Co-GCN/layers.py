import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, input_size, output_size, view_size=2, bias=True,indep_weights=False):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.indep_weights = indep_weights
        self.view_size = view_size

        self.weight = Parameter(torch.FloatTensor(input_size, output_size))
        self.pi = Parameter(torch.FloatTensor(view_size))

        if self.indep_weights:
            self.weight_global = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_leaf = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_or = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_and = Parameter(torch.FloatTensor(input_size, output_size))
            self.weight_not = Parameter(torch.FloatTensor(input_size, output_size))
        else:
            self.register_parameter('weight_global', None)
            self.register_parameter('weight_leaf', None)
            self.register_parameter('weight_or', None)
            self.register_parameter('weight_and', None)
            self.register_parameter('weight_not', None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.pi.data.uniform_(-stdv, stdv)

        if self.indep_weights:
            self.weight_global.data.uniform_(-stdv, stdv)
            self.weight_leaf.data.uniform_(-stdv, stdv)
            self.weight_or.data.uniform_(-stdv, stdv)
            self.weight_and.data.uniform_(-stdv, stdv)
            self.weight_not.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = None
        # print(input)
        # print(adj)
        for i in range(self.view_size):
            # print(self.pi[i])
            # print(adj[i])
            temp = self.pi[i] * adj[i]
            # print(temp)
            if support is None:
                support = temp
            else:
                support += temp
        # print(support.shape)
        # print(input.shape)
        support = torch.mm(support, input)
        # output = torch.spmm(adj, support)
        # print(support.shape)
        # print(self.weight.shape)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
