import os
import time
import torch
import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from layers import *
from models import *
from preprocessing import *

from convert_datasets_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score

from copy import deepcopy
import random


def pretrain(num_negs):

    features, hyperedge_index = data.x, data.edge_index
    # num_nodes, num_edges = data.num_nodes, data.num_edges
    cidx = hyperedge_index[1].min()
    hyperedge_index[1] -= cidx  # make sure we do not waste memory
    num_nodes = int(hyperedge_index[0].max()) + 1
    num_edges = int(hyperedge_index[1].max()) + 1
    model.train()
    model.zero_grad()
    hyperedge_index1 = drop_incidence(hyperedge_index, args.pretrain_drop_incidence_rate)
    hyperedge_index2 = drop_incidence(hyperedge_index, args.pretrain_drop_incidence_rate)
    x1 = drop_features(features, args.pretrain_drop_feature_rate)
    x1Shape = x1.shape
    x2 = drop_features(features, args.pretrain_drop_feature_rate)
    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1 & node_mask2
    edge_mask = edge_mask1 & edge_mask2
    # edgeMaskShape = edge_mask.shape
    device = data.x.device
    data1 = data.clone().to(device)
    data1.x = x1
    data1.edge_index = hyperedge_index1
    data1 = norm_contruction(data1, option=args.normtype).to(device)

    data2 = data.clone()
    data2.x = x2
    data2.edge_index = hyperedge_index2
    data2 = norm_contruction(data1, option=args.normtype).to(device)

    out_score_logits1, e1, n1, weight_tuple1 = model.forward(data1, edge_weight=None, pretrain=True, num_nodes=num_nodes, num_edges=num_edges)
    out_score_logits2, e2, n2, weight_tuple2 = model.forward(data2, edge_weight=None, pretrain=True, num_nodes=num_nodes, num_edges=num_edges)
    out1 = torch.sigmoid(out_score_logits1)
    out2 = torch.sigmoid(out_score_logits2)
    e1, e2 = torch.sigmoid(e1), torch.sigmoid(e2)
    n1, n2 = torch.sigmoid(n1), torch.sigmoid(n2)

    # fulle1 = torch.zeros((len(edge_mask1), *e1.shape[1:]), dtype=e1.dtype, device=e1.device)
    # r = 0
    # for i in range(len(edge_mask1)):
    #     if edge_mask1[i].item():
    #         fulle1[i] = e1[r]
    #         r += 1
    # fulle2 = torch.zeros((len(edge_mask2), *e2.shape[1:]), dtype=e1.dtype, device=e1.device)
    # r = 0
    # for i in range(len(edge_mask2)):
    #     if edge_mask2[i].item():
    #         fulle2[i] = e2[r]
    #         r += 1
    #
    # fulln1 = torch.zeros((len(node_mask1), *n1.shape[1:]), dtype=n1.dtype, device=n1.device)
    # r = 0
    # for i in range(len(node_mask1)):
    #     if node_mask1[i].item():
    #         fulln1[i] = n1[r]
    #         r += 1
    #
    # fulln2 = torch.zeros((len(node_mask2), *n2.shape[1:]), dtype=n2.dtype, device=n2.device)
    # r = 0
    # for i in range(len(node_mask2)):
    #     if node_mask2[i].item():
    #         fulln2[i] = n2[r]
    #         r += 1
    new_edge_mask1 = edge_mask[edge_mask1]
    new_edge_mask2 = edge_mask[edge_mask2]
    loss_n = model.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_batch_size, num_negs=num_negs)
    loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                    batch_size=args.pretrain_batch_size,
                                    num_negs=num_negs)
    masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
    masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
    loss_m1 = model.membership_level_loss(n1, e2[edge_mask2], masked_index2, args.pretrain_tau_m,
                                          batch_size=args.pretrain_batch_size)
    loss_m2 = model.membership_level_loss(n2, e1[edge_mask1], masked_index1, args.pretrain_tau_m,
                                          batch_size=args.pretrain_batch_size)
    loss_m = (loss_m1 + loss_m2) * 0.5
    loss = loss_n + args.pretrain_w_g * loss_g + args.pretrain_w_m * loss_m
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    model_optimizer.step()
    return loss.item()


def drop_features(x: Tensor, p: float):
    device = x.device  # Get device from input tensor
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=device).uniform_(0, 1) < p
    x = x.clone().to(device)  # Ensure x is on the same device
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges), device=device).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index



def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges), device=device).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    device = hyperedge_index.device  # Get device from input tensor
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1]).to(device)
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    device = hyperedge_indexs[0].device  # Get device from input tensor
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges).to(device)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool, device=device)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool, device=device)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    device = hyperedge_index.device  # Get device from input tensor
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges), device=device).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)

    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)


def parse_method(args, data):
    model = None
    if args.dname in ['mimic3', 'cradle', 'promote']:
        model = SetGNN(args, data)
    return model


# random seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    valid_acc_gf = valid_auc_gf = valid_aupr_gf = valid_f1_macro_gf = \
        test_acc_gf = test_auc_gf = test_aupr_gf = test_f1_macro_gf = \
        valid_acc_gcf = valid_auc_gcf = valid_aupr_gcf = valid_f1_macro_gcf = \
        test_acc_gcf = test_auc_gcf = test_aupr_gcf = test_f1_macro_gcf = 0

    model.eval()

    # use original graph (G)
    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)

    valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = eval_func(
        data.y[split_idx['valid']], out_g[split_idx['valid']],
        epoch, method, dname, args, mode='dev_g', threshold=args.threshold)
    test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = eval_func(data.y[split_idx['test']],
                                                                     out_g[split_idx['test']],
                                                                     epoch, method, dname, args,
                                                                     mode='test_g',
                                                                     threshold=args.threshold)

    if args.vanilla:
        edge_index = weight_tuple[0]
        edge_weight = weight_tuple[1].reshape(-1)
        # num_hyperedges = data.num_hyperedges[0]
        num_hyperedges = data.num_hyperedges
        # if epoch == args.epochs - 1:
        # get_subset_ranking(edge_weight, edge_index, num_hyperedges, args)

    else:
        # get the edge weight
        view_learner.eval()
        weight_logits = view_learner(data, device)

        # gumbel softmax
        # temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + weight_logits) / args.temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        # use factual graph (G')
        out_score_gf_logits, _, _, _ = model(data, edge_weight=aug_edge_weight)  # use augmented graph
        out_gf = torch.sigmoid(out_score_gf_logits)

        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf = eval_func(
            data.y[split_idx['valid']],
            out_gf[split_idx['valid']],
            epoch, method, dname, args, mode='dev_gf', threshold=args.threshold)
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf = eval_func(
            data.y[split_idx['test']], out_gf[split_idx['test']],
            epoch, method, dname, args, mode='test_gf', threshold=args.threshold)

        # use counterfactual graph (G-G')
        out_score_gcf_logits, _, _, _ = model(data, edge_weight=1 - aug_edge_weight)  # use augmented graph
        out_gcf = torch.sigmoid(out_score_gcf_logits)

        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf = eval_func(
            data.y[split_idx['valid']],
            out_gcf[split_idx['valid']],
            epoch, method, dname, args,
            mode='dev_gcf', threshold=args.threshold)
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = eval_func(
            data.y[split_idx['test']], out_gcf[split_idx['test']],
            epoch, method, dname, args,
            mode='test_gcf', threshold=args.threshold)

        if epoch == args.epochs - 1:
            get_subset_ranking(aug_edge_weight, data.edge_index, data.num_hyperedges, args)

    return valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
        test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf


def get_subset_ranking(edge_weight, edge_index, num_hyperedges, args):
    edge_index_clone = edge_index.clone().detach().to('cpu').numpy()
    edge_weight_clone = edge_weight.reshape(1, -1).clone().detach().to('cpu').numpy()
    index_weight_concat = np.concatenate((edge_index_clone, edge_weight_clone), axis=0)

    index_weight_concat = index_weight_concat[:, index_weight_concat[2, :].argsort()[::-1]]

    edge_dict = {}
    for i in range(num_hyperedges):
        edge_dict[i] = []
    for i in tqdm(range(index_weight_concat.shape[1])):
        if index_weight_concat[1][i] < num_hyperedges:  # self loop
            edge_dict[index_weight_concat[1][i]].append(index_weight_concat[0][i])
    sorted_edge_dict = dict(sorted(edge_dict.items()))

    vanilla = ""
    if args.vanilla: vanilla = "_vanilla"
    with open(f"outputs/deleted_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_del, \
            open(f"outputs/remained_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_rem:
        for hyperedge in list(sorted_edge_dict.values()):
            rem_size = int(len(hyperedge) * args.remain_percentage)
            if rem_size < 5 and len(hyperedge) >= 5:
                rem_size = 5
            elif rem_size < 5 and len(hyperedge) < 5:
                rem_size = len(hyperedge)
            remain = [str(int(x)) for x in hyperedge[:rem_size]]
            f_rem.write(",".join(remain))
            f_rem.write('\n')
            delete = [str(int(x)) for x in hyperedge[rem_size:]]
            f_del.write(",".join(delete))
            f_del.write('\n')


def eval_mimic3(y_true, y_pred, epoch, method, dname, args, mode='dev', threshold=0.5):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)

    total_acc = []
    total_f1 = []
    for i in range(args.num_labels):
        correct = (pred[:, i] == y_true[:, i])
        accuracy = correct.sum() / correct.size
        total_acc.append(accuracy)
        f1_macro = f1_score(y_true[:, i], pred[:, i], average='macro')
        total_f1.append(f1_macro)

    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true, pred, average='macro')

    total_auc = []
    for i in range(args.num_labels):
        roc_auc = roc_auc_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_auc.append(roc_auc)

    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))

    total_aupr = []
    for i in range(args.num_labels):
        aupr = average_precision_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_aupr.append(aupr)
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    import csv
    with open(f'outputs/mimic3_{mode}_{method}.csv', 'a+', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Epoch", "Phenotype", "acc", "auc", 'aupr', 'f1'])
        for i, (acc_, auc_, aupr_, f1_) in enumerate(zip(total_acc, total_auc, total_aupr, total_f1)):
            write_lst = [epoch, f"Phenetype {i}", acc_, auc_, aupr_, f1_]
            writer.writerow(write_lst)

    return accuracy, roc_auc, aupr, f1_macro


def eval_cradle(y_true, y_pred, epoch, method, dname, args, mode='dev', threshold=0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true.reshape(-1), pred.reshape(-1), average="macro")
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    return accuracy, roc_auc, aupr, f1_macro


if __name__ == '__main__':
    os.chdir('/local/scratch/rwan388/benchmark/hypehr/src')  # working dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='promote')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--view_lr', default=1e-2, type=float)
    parser.add_argument('--view_wd', default=1e-3, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=2, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=48,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_labels', default=1, type=int)  # set the default for now
    parser.add_argument('--num_nodes', default=2653, type=int)  # 7423 for mimic and 12725 for cradle
    # 'all' means all samples have labels, otherwise it indicates the first [num_labeled_data] rows that have the labels
    parser.add_argument('--num_labeled_data', default='all', type=str)
    parser.add_argument('--feature_dim', default=128, type=int)  # feature dim of learnable node feat
    parser.add_argument('--LearnFeat', action='store_true')
    # whether the he contain self node or not
    parser.add_argument('--PMA', action='store_true')
    #     Args for Attentions
    parser.add_argument('--heads', default=4, type=int)  # Placeholder
    parser.add_argument('--output_ ', default=1, type=int)  # Placeholder

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--view_alpha', type=float, default=0.5)
    parser.add_argument('--view_lambda', type=float, default=5)
    parser.add_argument('--model_lambda', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1)  # 0.5 | 5; temperature for gumbel softmax

    parser.add_argument('--vanilla', action='store_true', default=True)
    parser.add_argument('--remain_percentage', default=0.3, type=float)
    parser.add_argument('--rand_seed', default=0, type=int)
    parser.add_argument('--method', default='AllSetTransformer', type=str)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_epoch', default=100, type=int)
    parser.add_argument('--pretrain_weight_decay', default=1.0e-05, type=float)
    parser.add_argument('--pretrain_lr', default=1e-3, type=float)
    parser.add_argument('--pretrain_drop_incidence_rate', default=0.3, type=float)
    parser.add_argument('--pretrain_drop_feature_rate', default=0.3, type=float)
    parser.add_argument('--pretrain_tau_n', default=0.5, type=float)
    parser.add_argument('--pretrain_tau_g', default=0.5, type=float)
    parser.add_argument('--pretrain_tau_m', default=1.0, type=float)
    parser.add_argument('--pretrain_w_g', default=4, type=float)
    parser.add_argument('--pretrain_w_m', default=1, type=float)
    parser.add_argument('--pretrain_batch_size', default=100, type=int)

    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=False)

    args = parser.parse_args()

    seed_everything(args.rand_seed)

    existing_dataset = ['mimic3', 'cradle', 'promote']

    synthetic_list = ['mimic3', 'cradle', 'promote']

    dname = args.dname
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes)
    data = dataset.data
    args.num_features = dataset.num_features
    if args.dname in ['mimic3', 'cradle', 'promote']:
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = norm_contruction(data, option=args.normtype)

    model = parse_method(args, data)
    view_learner = ViewLearner(parse_method(args, data), args.MLP_hidden)
    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # print("device",device);


    #     Get splits
    split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop,
                                    rand_seed=args.rand_seed)

    train_idx = split_idx['train'].to(device)

    model, view_learner, data = model.to(device), view_learner.to(device), data.to(device)

    criterion = nn.BCELoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

    with open(f'../data/raw_data/{args.dname}/hyperedges-{args.dname}.txt', 'r') as f:
        total_edges = []
        maxlen = 0
        for lines in f:
            line = lines.strip().split(',')
            line = list(map(int, line))
            if len(line) > maxlen:
                maxlen = len(line)
            total_edges.append(line)
        total_edges_padded = []
        for edge in total_edges:
            total_edges_padded.append(edge + [-1] * (maxlen - len(edge)))

    if args.num_labeled_data != 'all':
        N = int(args.num_labeled_data)  # the first x visits have labels
    elif args.num_labeled_data == 'all':
        N = len(total_edges_padded)  # all the samples in cradle have labels
    train_num = int(N * args.train_prop)
    valid_num = int(N * args.valid_prop)
    train_input = torch.LongTensor(total_edges_padded[:train_num]).to(device)
    dev_input = torch.LongTensor(total_edges_padded[train_num:train_num + valid_num]).to(device)
    test_input = torch.LongTensor(total_edges_padded[train_num + valid_num:N]).to(device)

    edge_id_dict = None
    with torch.autograd.set_detect_anomaly(True):
        if (args.pretrain):
            for epoch in trange(args.pretrain_epoch, desc='Epoch'):
                if epoch==4:
                    print("check")
                loss = pretrain(num_negs=None)
        for epoch in trange(args.epochs):
            if args.vanilla:  # VANILLA - Use attention weight to get an important set for each encounter
                model.train()
                model.zero_grad()

                out_score_logits, _, _, weight_tuple = model(data)
                out = torch.sigmoid(out_score_logits)

                model_loss = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
                    weight_tuple[1].reshape(-1))
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()
            else:  # CACHE
                if (epoch + 1) % 50 == 0:
                    args.view_lambda *= 0.5
                """STEP ONE - TRAIN THE LEARNER"""
                view_learner.train()
                view_learner.zero_grad()
                model.eval()

                out_score_logits, out_edge_feat, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                weight_logits = view_learner(data, device)

                # gumbel softmax
                # temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                gate_inputs = gate_inputs.to(device)
                gate_inputs = (gate_inputs + weight_logits) / args.temperature
                aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                # factual prediction
                out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                out_f = torch.sigmoid(out_score_f_logits)

                # regularization - not to drop too many edges
                edge_dropout_prob = 1 - aug_edge_weight
                reg = torch.mean(edge_dropout_prob)

                # counterfactual prediction
                out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                out_cf = torch.sigmoid(out_score_cf_logits)

                # factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = 1
                coef[out < 0.5] = -1
                loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                # counterfactual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = -1
                coef[out < 0.5] = 1
                loss_cf = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                # factual and counterfactual view loss
                loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                view_loss = loss + args.view_lambda * torch.mean(aug_edge_weight)
                view_loss.backward()
                torch.nn.utils.clip_grad_norm_(view_learner.parameters(), 1)
                view_optimizer.step()

                """STEP TWO - TRAIN THE MAIN MODEL"""
                model.train()
                model.zero_grad()
                view_learner.eval()

                out_score_logits, out_edge_feat, _, _ = model(data)
                out = torch.sigmoid(out_score_logits)

                # learn the edge weight (augmentation policy)
                weight_logits = view_learner(data, device)

                # gumbel softmax
                # temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                gate_inputs = gate_inputs.to(device)
                gate_inputs = (gate_inputs + weight_logits) / args.temperature
                aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                # factual prediction
                out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                out_f = torch.sigmoid(out_score_f_logits)

                # counterfactual prediction
                edge_dropout_prob = 1 - aug_edge_weight
                out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                out_cf = torch.sigmoid(out_score_cf_logits)

                # factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = 1
                coef[out < 0.5] = -1
                loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                # counter factual loss
                coef = out.detach().clone()
                coef[out >= 0.5] = -1
                coef[out < 0.5] = 1
                loss_cf = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                # factual and counterfactual view loss
                loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                model_loss = criterion(out[train_idx], data.y[train_idx]) + args.model_lambda * loss
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()

            if dname in ['mimic3']:
                eval_function = eval_mimic3
            elif dname in ['cradle', 'promote']:
                eval_function = eval_cradle
            valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
                test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
                valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
                test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
                valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
                test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = \
                evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname,
                         args)

            fname_dev = ''
            fname_test = ''
            vanilla = ""
            if args.vanilla: vanilla = "_vanilla"
            if dname == 'mimic3':
                fname_dev = f'outputs/mimic3_dev_{args.method}{vanilla}.txt'
                fname_test = f'outputs/mimic3_test_{args.method}{vanilla}.txt'
            elif dname == 'cradle':
                fname_dev = f'outputs/cradle_dev_{args.method}{vanilla}.txt'
                fname_test = f'outputs/cradle_test_{args.method}{vanilla}.txt'
            elif dname == 'promote':
                fname_dev = f'outputs/promote_dev_{args.method}{vanilla}.txt'
                fname_test = f'outputs/promote_test_{args.method}{vanilla}.txt'
            # dev set
            with open(fname_dev, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n '
                    .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                            args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, valid_acc_g,
                            valid_auc_g, valid_aupr_g, valid_f1_macro_g, valid_acc_gf, valid_auc_gf, valid_aupr_gf,
                            valid_f1_macro_gf,
                            valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf))
            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n'
                    .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                            args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, test_acc_g,
                            test_auc_g, test_aupr_g, test_f1_macro_g, test_acc_gf, test_auc_gf, test_aupr_gf,
                            test_f1_macro_gf, test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf))

    print('All done! Exit python code')
    quit()

