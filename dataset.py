import numpy as np
import pandas as pd
import os.path as osp

def load_dataset(path='../data/raw_data/', dataset='promote', node_feature_path="../data/promote/node-embeddings-promote.npy", num_node=2653, num_labeled_data='all', num_labels=1):
    # first load edge labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'edge-labels-{dataset}.txt'), sep=',', header=None)
    print(f"load label information from 'edge-labels-{dataset}.txt'. ")
    num_edges = df_labels.shape[0]
    labels = df_labels.values
    
    
    # load features.npy
    features = np.load(node_feature_path)
    
    print(f"load features from {node_feature_path}. ")
    if num_labeled_data != 'all':
        features = features[:int(num_labeled_data)]
        labels = labels[:int(num_labeled_data)]
    labels = np.array(labels)
    assert labels.shape[1] == num_labels
    assert features.shape[0] == labels.shape[0]
    
    return features, labels

