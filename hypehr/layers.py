#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional


# This part is for PMA.
# Modified from GATConv in pyg.
# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class PMA(MessagePassing):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        #         kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'
        #         self.input_seed = input_seed

        #         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
        #         Now, no seed as input. Directly learn the importance weights alpha_ij.
        #         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads * self.hidden,
                       hidden_channels=self.heads * self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None', )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        #         if bias and concat:
        #             self.bias = Parameter(torch.Tensor(heads * out_channels))
        #         elif bias and not concat:
        #             self.bias = Parameter(torch.Tensor(out_channels))
        #         else:

        #         Always no bias! (For now)
        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        #         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)


    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None, edge_weight=None, max_index=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1))
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
        device = x.device
        edge_index = edge_index.to(device)
        x_V = x_V.to(device)
        alpha_r = alpha_r.to(device)
        edge_weight = edge_weight.to(device)
        if max_index is not None:
            out = self.propagate(edge_index.clone().to(device), x=x_V,
                                 alpha=alpha_r, aggr=self.aggr, edge_weight=edge_weight,
                                 max_index=max_index, size=size)
        else:
            out = self.propagate(edge_index.clone().to(device), x=x_V,
                                 alpha=alpha_r, aggr=self.aggr, edge_weight=edge_weight,
                                 size=size,max_index=None)
        alpha = self._alpha
        self._alpha = None

        #         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
        #         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j, edge_weight, max_index=None):
        #         ipdb.set_trace()
        if max_index is None :
            max_index = index.max()+1
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1) #instead of index.max I think I should passed edge and node here!
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print(x_j.device)
        # print(edge_weight.device)
        # print(alpha.device)
        res=x_j * alpha.unsqueeze(-1) if edge_weight is None else x_j * alpha.unsqueeze(-1) * edge_weight.view(-1, 1, 1)

        return x_j * alpha.unsqueeze(-1) if edge_weight is None else x_j * alpha.unsqueeze(-1) * edge_weight.view(-1, 1,
                                                                                                                  1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add',max_index=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggr is None:
            raise ValueError("aggr was not passed!")
        if max_index is not None:
            dim_size = max_index if max_index > index.max().item() + 1 else index.max().item() + 1
        else:
            dim_size = index.max().item() + 1
        aggregated = scatter(inputs, index, dim=self.node_dim, reduce=aggr,dim_size=dim_size)
        # aggregated = scatter(inputs, index, dim=self.node_dim, reduce=aggr)
        return aggregated

        # If `dim_size` is specified, ensure the output tensor has the correct number of rows
        # if dim_size is not None:
        #     # Create a tensor of zeros with the desired size
        #     zeros_tensor = torch.zeros((dim_size, *aggregated.shape[1:]), device=inputs.device)
        #
        #     # Fill in the aggregated values into the zeros tensor
        #     zeros_tensor[:aggregated.size(0)] = aggregated
        #
        #     # Return the tensor with zeros filled for nodes without incoming edges
        #     return zeros_tensor
        # else:
        #     # Return the aggregated tensor directly
        #     return aggregated

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True
                 ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    #         self.bn = nn.BatchNorm1d(dec_hid_dim)
    #         self.dropout = dropout
    #         self.Prop = S2SProp()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is 'Identity'):
                self.f_dec.reset_parameters()

    #         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add', edge_weight=None, max_index=None):
        """
        input -> MLP -> Prop
        """
        device = x.device  # Get the device from the input tensor
        # print('edge_index',edge_index.device)
        # print('device',device)

        # Move other tensors to the same device as the input tensor
        edge_index = edge_index.to(device)
        if norm is not None:
            norm = norm.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        weight_tuple = None
        if self.attention:
            # print('ATTENTION')
            # if edge_weight is not None:
            #     print(edge_weight)
            self.prop.to(device)
            x, weight_tuple = self.prop(x, edge_index, edge_weight=edge_weight, return_attention_weights=True, max_index=max_index)
            # print(weight_tuple[0].shape, weight_tuple[1].shape)
            # print(weight_tuple[0], weight_tuple[1])
            # exit()
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr, edge_weight=edge_weight, max_index=max_index)
            x = F.relu(self.f_dec(x))

        return x, weight_tuple

    def message(self, x_j, norm, edge_weight):  # Add edge weight
        # Ensure tensors are on the same device
        device = x_j.device
        norm = norm.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        return norm.view(-1, 1) * x_j if edge_weight is None else norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add'):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        device = inputs.device
        index = index.to(device)
        if aggr is None:
            raise ValueError("aggr was not passed!")
        aggregated = scatter(inputs, index, dim=self.node_dim, reduce=aggr, dim_size=dim_size)
        if dim_size is not None:
            zeros_tensor = torch.zeros((dim_size, aggregated.size(1)), device=inputs.device)
            zeros_tensor[:aggregated.size(0)] = aggregated
            return zeros_tensor
        else:
            # Return the aggregated tensor directly
            return aggregated
        # return scatter(inputs, index, dim=self.node_dim, reduce=aggr)


