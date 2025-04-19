#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax

import numpy as np


class SetGNN(nn.Module):
    def __init__(self, args, data, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = True
        self.LearnFeat = args.LearnFeat

        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnFeat:
            self.x = Parameter(data.x, requires_grad=True)

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_labels,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.feature_dim,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            for i in range(self.All_num_layers):
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                if i < self.All_num_layers - 1:
                    self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.classifier = MLP(
                # in_channels=args.MLP_hidden,
                in_channels=args.MLP_hidden * (args.All_num_layers + 1),
                hidden_channels=args.Classifier_hidden,
                out_channels=args.num_labels,
                num_layers=args.Classifier_num_layers,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False)
            self.disc = nn.Bilinear(args.MLP_hidden, args.MLP_hidden, 1)

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data, edge_weight=None, pretrain=False,num_nodes=None, num_edges=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        device = data.x.device
        # print('model->device',device)
        # print('model-edge_index',edge_index.device)
        if self.LearnFeat:
            x = self.x
        #
        data = data.to(device)
        edge_index = edge_index.to(device)
        if norm is not None:
            norm = norm.to(device)
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        vec = []
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            if pretrain:
                x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr,
                                                   edge_weight=edge_weight, max_index=num_edges)  # 这里传入限制的边和node
            else:
                x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr, edge_weight=edge_weight)#这里传入限制的边和node
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if pretrain:
                x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, max_index=num_nodes)
            else:
                x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if pretrain:
            x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight,max_index=num_edges)
        else:
            x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
        # PairNorm
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        x = x[:data.y.shape[0], :]
        edge_score = self.classifier(x)

        return edge_score, edge_feat, node_feat, weight_tuple

    def f(self, x, tau):
        if tau == 0:
            raise ValueError("Tau cannot be zero in the exponential function.")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input 'x' contains NaN or infinite values.")
        return torch.exp(x / tau)

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        if z1.shape[1] != z2.shape[1]:
            raise ValueError("Mismatched shapes between z1 and z2 in cosine_similarity.")
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):

        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        print(f"h1 shape: {h1.shape}, h2 shape: {h2.shape}")
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        # print(f"n1_filtered shape: {n1.shape}")
        # print(f"n2_filtered shape: {n2.shape}")
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        if batch_size is None:
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            # by default selected k(negative sample per positive sample) = 1?
            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]
                if len(mask) > 0:             
                    pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)
                    if pos.shape != ():
                        neg_n = self.f(
                            self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]), tau)
                        neg_e = self.f(
                            self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)

                        # Append results only if mask > 0 and pos is not 0-dimensional
                        aggr_pos.append(pos)
                        aggr_neg_n.append(neg_n)
                        aggr_neg_e.append(neg_e)

            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)
            # aggr_pos = self.concatenate_tensors(aggr_pos)
            # aggr_neg_n = self.concatenate_tensors(aggr_neg_n)
            # aggr_neg_e = self.concatenate_tensors(aggr_neg_e)

            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()
        return loss

    def concatenate_tensors(self, tensor_list):
        """
        Concatenate tensors from a list while handling zero-dimensional tensors.
        """
        # Initialize an empty list for concatenation
        tensors_to_concatenate = []
        
        # Iterate through the tensor list
        for tensor in tensor_list:
            # If the tensor is not zero-dimensional, add it to the list for concatenation
            if tensor.dim() > 0:
                tensors_to_concatenate.append(tensor)
            else:
                # Skip zero-dimensional tensors or replace them with an empty tensor of appropriate shape
                # Alternatively, you can choose to log a warning or handle the zero-dimensional tensor in another way
                pass
        
        # If there are tensors to concatenate, perform the concatenation
        if tensors_to_concatenate:
            concatenated_tensor = torch.concat(tensors_to_concatenate)
            return concatenated_tensor
        
        # If no tensors to concatenate, return an empty tensor of appropriate type and device
        else:
            # Return an empty tensor with the same dtype and device as the first tensor in the original list
            if tensor_list:
                dtype = tensor_list[0].dtype
                device = tensor_list[0].device
                return torch.tensor([], dtype=dtype, device=device)
            else:
                return torch.tensor([])



class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, input_dim, viewer_hidden_dim=64):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = input_dim

        self.mlp_edge_model = nn.Sequential(
            Linear(self.input_dim * 2, viewer_hidden_dim),
            nn.ReLU(),
            Linear(viewer_hidden_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, device):

        _, edge_feat, node_feat, _ = self.encoder(data.clone())

        totedges = data.totedges
        # num_hyperedges = data.num_hyperedges[0]
        num_hyperedges = data.num_hyperedges
        num_self_loop = totedges - num_hyperedges
        edge_index = data.edge_index.clone()
        # num_self_loop_clone = num_self_loop.clone()
        num_self_loop_clone = int(num_self_loop)
        node, edge = edge_index[:, :-num_self_loop_clone][0], edge_index[:, :-num_self_loop_clone][1]
        emb_node = node_feat[node]
        emb_edge = edge_feat[edge]

        total_emb = torch.cat([emb_node, emb_edge], 1)
        edge_weight = self.mlp_edge_model(total_emb)

        self_loop_weight = np.ones(shape=(num_self_loop_clone, 1)) * 10.0
        self_loop_weight = torch.FloatTensor(self_loop_weight).to(device)
        weight_logits = torch.cat([edge_weight, self_loop_weight], 0)

        return weight_logits
