
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# For Graph Neural Networks

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias = Parameter(torch.zeros(out_features,dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('leaky_relu',0.01))

    def forward(self, input, adj):
        # single sample in batch needed !!
        support = torch.mm(input[0], self.weight)
        output = torch.spmm(adj, support)
        # # batch
        # support = einsum(input,self.weight,'batch nodes features, features out_features -> batch nodes out_features')
        # output = einsum(adj,support,'i nodes, batch nodes features -> batch i features')
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'