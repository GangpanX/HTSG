"""Data loading."""
import os
import pickle as pkl
import sys
from torch.nn.parameter import Parameter
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn


def load_data(args, datapath):
    data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    data['adj_train_norm'], data['text_features'], data['image_features'] = process(
        data['adj_train'], data['text_features'], data['image_features'], args.normalize_adj, args.normalize_feats
    )
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, text_features, image_features, normalize_adj, normalize_feats):
    if sp.isspmatrix(text_features):
        text_features = np.array(text_features.todense())
    if sp.isspmatrix(image_features):
        image_features = np.array(image_features.todense())
    if normalize_feats:
        text_features = normalize(text_features)
        image_features = normalize(image_features)

    text_features = torch.Tensor(text_features)
    image_features = torch.Tensor(image_features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, text_features, image_features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset == 'twitter':
        adj, text_features, image_features, labels = load_data_twitter(use_feats, data_path)
        val_prop, test_prop = 0.10, 0.20
    elif dataset == 'weibo':
        adj, text_features, image_features, labels = load_data_weibo(use_feats, data_path)
        val_prop, test_prop = 0.10, 0.20
    elif dataset == 'pheme':
        adj, text_features, image_features, labels = load_data_pheme(use_feats, data_path)
        val_prop, test_prop = 0.10, 0.20
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'text_features': text_features, 'image_features': image_features, 'labels': labels,
            'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


def load_data_twitter(use_feats, data_path):
    adj = np.load(os.path.join(data_path, "twitter_A.npy"))
    text_features = np.load(os.path.join(data_path, "txt_vector.npy"))
    image_features = np.load(os.path.join(data_path, "image_vector.npy"))
    label = np.load(os.path.join(data_path, "label.npy"))

    if not use_feats:
        text_features = sp.eye(adj.shape[0])
        image_features = sp.eye(adj.shape[0])
    adj = sp.csr_matrix(adj)
    return adj, text_features, image_features, label


def load_data_weibo(use_feats, data_path):
    adj = np.load(os.path.join(data_path, "weibo_A.npy"))
    text_features = np.load(os.path.join(data_path, "txt_vector.npy"))
    image_features = np.load(os.path.join(data_path, "image_vector.npy"))
    label = np.load(os.path.join(data_path, "label.npy"))

    if not use_feats:
        text_features = sp.eye(adj.shape[0])
        image_features = sp.eye(adj.shape[0])
    adj = sp.csr_matrix(adj)
    return adj, text_features, image_features, label


def load_data_pheme(use_feats, data_path):
    adj = np.load(os.path.join(data_path, "pheme_A.npy"))
    text_features = np.load(os.path.join(data_path, "txt_vector.npy"))
    image_features = np.load(os.path.join(data_path, "image_vector.npy"))
    label = np.load(os.path.join(data_path, "label.npy"))

    if not use_feats:
        text_features = sp.eye(adj.shape[0])
        image_features = sp.eye(adj.shape[0])
    adj = sp.csr_matrix(adj)
    return adj, text_features, image_features, label
