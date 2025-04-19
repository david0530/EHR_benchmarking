
import os
import time
import torch
import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp

from tqdm import tqdm, trange

# from layers import *
# from models import *
from preprocessing import *
from dataset import *

from convert_datasets_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

from copy import deepcopy
import random



# random seed 
def seed_everything(seed=0):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   


def eval_mimic3(y_true, y_pred, args, mode='dev', threshold=0.5):
    acc_list = []
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
    print(f'ACC:{accuracy}, AUC:{roc_auc}, AUPR:{aupr}, F1_macro:{f1_macro}')

    # import csv
    # with open(f'outputs/mimic3_{mode}_{method}.csv', 'a+', encoding='utf-8') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow(["Epoch", "Phenotype", "acc", "auc", 'aupr', 'f1'])
    #     for i, (acc_, auc_, aupr_, f1_) in enumerate(zip(total_acc, total_auc, total_aupr, total_f1)):
    #         write_lst = [f"Phenetype {i}", acc_, auc_, aupr_, f1_]
    #         writer.writerow(write_lst)

    return accuracy, roc_auc, aupr, f1_macro


def eval_cradle(y_true, y_pred, mode='dev', threshold=0.5):
    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true.reshape(-1), pred.reshape(-1), average="macro")
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))
    print(f'ACC:{accuracy}, AUC:{roc_auc}, AUPR:{aupr}, F1_macro:{f1_macro}')
    return accuracy, roc_auc, aupr, f1_macro


if __name__ == '__main__':
    os.chdir('/local/scratch/rwan388/benchmark/sklearn/src')  # need to modify
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='promote')
    parser.add_argument('--num_labels', default=1, type=int)  # 1 for cradle and promote, 25 for mimic
    parser.add_argument('--num_nodes', default=2653, type=int)  # 7423 for mimic and 12725 for cradle and 2653 for promote
    # 'all' means all samples have labels, otherwise it indicates the first [num_labeled_data] rows that have the labels
    parser.add_argument('--node_feature', default='avgnode', type=str)  
    # avgnode or multihot
    parser.add_argument('--num_labeled_data', default='all', type=str)
    # 12353 for mimic, 'all' for cradle and promote
    parser.add_argument('--rand_seed', default=0, type=int)   # 0/1/2/3/4
    parser.add_argument('--random_split', default=True, type=bool)
    parser.add_argument('--logistic_regression', default=False, type=bool)
    parser.add_argument('--svm', default=False, type=bool)
    parser.add_argument('--mlp', default=False, type=bool)
    parser.add_argument('--random_forest', default=False, type=bool)
    parser.add_argument('--naive_bayes', default=False, type=bool)
    parser.add_argument('--xgboost', default=False, type=bool)
    args = parser.parse_args()
    seed_everything(args.rand_seed) 

    # load data
    p2raw = '../data/raw_data/'
    features, labels = load_dataset(path=p2raw, dataset=args.dname, node_feature_path=f'{p2raw}/{args.dname}/node-embeddings-{args.dname}_{args.node_feature}.npy', num_node=args.num_nodes, num_labels=args.num_labels, num_labeled_data = args.num_labeled_data)



    # Get splits
    split_idx = rand_train_test_idx(labels, train_prop=args.train_prop, valid_prop=args.valid_prop, rand_seed=args.rand_seed, random_split=args.random_split)
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    # train_idx = split_idx['train'][:100]
    # valid_idx = split_idx['valid'][:100]
    # test_idx = split_idx['test'][:100]
    
    # load model
    if args.logistic_regression:
        print('-'*10 + 'Logistic Regression' + '-'*10)
        if args.dname in ['cradle', 'promote']:
            # num_labels = 1
            labels = labels.flatten()   # pass 1d label to sklearn model
            model = LogisticRegression(max_iter=1000, random_state=args.rand_seed, C=1)
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and logistic regression method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and logistic regression method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=args.rand_seed, C=7))
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(f'Validation set results for {args.dname} and logistic regression method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and logistic regression method')
            eval_mimic3(labels[test_idx], test_prob, args)
    
    if args.svm:    # it takes a really long time to run svm
        print('-'*10 + 'SVM' + '-'*10)
        if args.dname in ['cradle', 'promote']:
            model = SVC(kernel='rbf', C=1, gamma='scale', coef0=0, random_state=args.rand_seed, probability=True)
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and SVM method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and SVM method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma='scale', coef0=0, random_state=args.rand_seed, probability=True))
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(model.get_params())
            print(f'Validation set results for {args.dname} and SVM method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and SVM method')
            eval_mimic3(labels[test_idx], test_prob, args)
    
    if args.mlp:
        print('-'*10 + 'MLP' + '-'*10)
 
        if args.dname in ['cradle', 'promote']:
            model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=1e-3, validation_fraction=0, max_iter=200, random_state=args.rand_seed)
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and MLP method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and MLP method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=1e-3, validation_fraction=0, max_iter=200, random_state=args.rand_seed))
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(model.get_params())
            print(f'Validation set results for {args.dname} and MLP method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and MLP method')
            eval_mimic3(labels[test_idx], test_prob, args)
        
    if args.random_forest:
        print('-'*10 + 'Random Forest' + '-'*10)
        if args.dname in ['cradle', 'promote']:
            model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=args.rand_seed)
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and random forest method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and random forest method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=args.rand_seed))
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(model.get_params())
            print(f'Validation set results for {args.dname} and random forest method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and random forest method')
            eval_mimic3(labels[test_idx], test_prob, args)
    
    if args.naive_bayes:
        print('-'*10 + 'Naive Bayes' + '-'*10)
        # model = BernoulliNB()
        if args.dname in ['cradle', 'promote']:
            model = GaussianNB()
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and naive bayes method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and naive bayes method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(GaussianNB())
            model.fit(features[train_idx], labels[train_idx])
            al_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(model.get_params())
            print(f'Validation set results for {args.dname} and naive bayes method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and naive bayes method')
            eval_mimic3(labels[test_idx], test_prob, args)

    if args.xgboost:
        print('-'*10 + 'XGBoost' + '-'*10) 
        if args.dname in ['cradle', 'promote']:
            model = XGBClassifier(n_estimators=100, max_depth=6, max_leaves=0, learning_rate=0.1, random_state=args.rand_seed)
            model.fit(features[train_idx], labels[train_idx])
            val_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])[:, 1]
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])[:, 1]
            print(model.get_params())
            print(f'Validation set results for {args.dname} and XGBoost method')
            eval_cradle(labels[valid_idx], val_prob, )
            print(f'Test set results for {args.dname} and XGBoost method')
            eval_cradle(labels[test_idx], test_prob, )
        if args.dname in ['mimic3']:
            model = OneVsRestClassifier(XGBClassifier(n_estimators=100, max_depth=6, max_leaves=0, learning_rate=0.1, random_state=args.rand_seed))
            model.fit(features[train_idx], labels[train_idx])
            al_pred = model.predict(features[valid_idx])
            val_prob = model.predict_proba(features[valid_idx])
            test_pred = model.predict(features[test_idx])
            test_prob = model.predict_proba(features[test_idx])
            print(model.get_params())
            print(f'Validation set results for {args.dname} and xgboost method')
            eval_mimic3(labels[valid_idx], val_prob, args)
            print(f'Test set results for {args.dname} and xgboost method')
            eval_mimic3(labels[test_idx], test_prob, args)
            
    print('All done! Exit python code')
    quit()
