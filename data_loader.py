import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import json
import pickle

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math_support import compute_qr_factorization
from math_support import convert_edge_index_to_list

import torch_geometric
from sklearn.decomposition import PCA

def convert_label_to_lprob(ylabel, eps=1.0e-6):
    npoint = ylabel.size
    nclass = np.int64(np.max(ylabel) + 1)
    ylprob = eps + np.zeros((npoint, nclass))
    for idx in range(npoint):
        ylprob[idx, ylabel[idx]] = 1.0 + eps - nclass * eps
    return np.log(ylprob)

def compute_user_item_matrix(edge_data):
    n = edge_data.shape[0]
    smat = np.zeros((n))
    uidx = np.zeros((n), dtype=np.int64)
    iidx = np.zeros((n), dtype=np.int64)
    for idx in range(n):
        smat[idx] = 1.0
        uidx[idx] = edge_data[idx, 0]
        iidx[idx] = edge_data[idx, 1]
    return (smat, uidx, iidx)


def compute_user_item_embedding(smat, uidx, iidx, embedding_dimension, niter=1000):
    print('compute_user_item_embedding')
    dimy = np.max(iidx) + 1
    umat, imat = compute_qr_factorization(smat, uidx, iidx, embedding_dimension, dimy, niter, eps=1.0e-6)
    umat = np.transpose(umat)
    imat = np.transpose(imat)
    return imat




def compute_weight_matrix(eps=1.0e-6):
    data = torch_geometric.datasets.CitationFull('./data_torch_geometric', 'dataset_name')[0]
    edge_data = np.transpose(data['edge_index'].cpu().detach().numpy().astype(np.int64))
    # node_data = np.loadtxt('./data/citeseer_nodes.csv', delimiter=',', skiprows=1).astype(np.int64)
    nuser = np.max(edge_data[:, 0]) + 1
    nitem = np.max(edge_data[:, 1]) + 1
    n = max(nuser, nitem)
    wmat = np.zeros((n, n))
    for idx in range(edge_data.shape[0]):
        idx0 = edge_data[idx, 0]
        idx1 = edge_data[idx, 1]
        wmat[idx0, idx1] = 1.0
    for idx in range(n):
        wsum = np.sum(wmat[idx, :])
        if wsum < eps:
            wmat[idx, idx] = 1.0
        else:
            wmat[idx, :] = wmat[idx, :] / np.sum(wmat[idx, :])
    return wmat



def perform_ttv_split(nsample, ftrain=0.6, fttest=0.2, fvalid=0.2):
    nvalid = np.int64(fvalid * nsample)
    nttest = np.int64(fttest * nsample)
    ntrain = nsample - nttest - nvalid
    perm = np.random.permutation(nsample)
    idx_train = perm[:ntrain]
    idx_ttest = perm[ntrain : (-nvalid)]
    idx_valid = perm[(-nvalid):]
    return (idx_train, idx_ttest, idx_valid)



def read_data(embedding_dimension=1, dataset_name='', eps=1.0e-6):
    data = torch_geometric.datasets.CitationFull('./data_torch_geometric', dataset_name)[0]
    xembed = data['x'].cpu().detach().numpy()
    eindex = data['edge_index'].cpu().detach().numpy().astype(np.int64)
    ylabel = data['y'].cpu().detach().numpy().astype(np.int64)
    pca = PCA(n_components=embedding_dimension)
    #xembed = pca.fit_transform(xembed)


    eglist = convert_edge_index_to_list(eindex)
    smat, uidx, iidx = compute_user_item_matrix(np.transpose(eindex))
    # xsvd = compute_user_item_embedding(smat, uidx, iidx, embedding_dimension, niter=1000)
    xsvd = xembed.copy()
    ylprob = convert_label_to_lprob(ylabel, eps=1.0e-10)
    return (xembed, eglist, ylabel, ylprob, xsvd)

