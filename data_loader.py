import numpy as np
from sklearn.preprocessing import StandardScaler
from math_support import compute_sparse_laplace_data
from math_support import compute_laplace_eigen_basis

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
from math_support import convert_edge_index_to_list

from torch_geometric.datasets import Planetoid, WikipediaNetwork
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
    return smat, uidx, iidx


def compute_weight_matrix(dataset_name, eps=1.0e-6):
    data = Planetoid('./data_torch_geometric', dataset_name)[0]
    edge_data = np.transpose(data.edge_index.cpu().detach().numpy().astype(np.int64))
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
            wmat[idx, :] = wmat[idx, :] / wsum
    return wmat


def perform_ttv_split(nsample, ftrain=0.6, fttest=0.2, fvalid=0.2):
    nvalid = np.int64(fvalid * nsample)
    nttest = np.int64(fttest * nsample)
    ntrain = nsample - nttest - nvalid
    perm = np.random.permutation(nsample)
    idx_train = perm[:ntrain]
    idx_ttest = perm[ntrain:(-nvalid)]
    idx_valid = perm[(-nvalid):]
    return idx_train, idx_ttest, idx_valid


def read_data(embedding_dimension=8, bundle_dimension=32, dataset_name='Cora',
              split_id=0, eps=1.0e-6):
    if dataset_name in ['Chameleon', 'Squirrel']:
        data = WikipediaNetwork(
            './data_torch_geometric',
            name=dataset_name.lower(),
            geom_gcn_preprocess=True
        )[0]
    else:
        data = Planetoid('./data_torch_geometric', dataset_name)[0]

    xembed = data.x.cpu().detach().numpy()
    eindex = data.edge_index.cpu().detach().numpy().astype(np.int64)
    ylabel = data.y.cpu().detach().numpy().astype(np.int64)

    scaler = StandardScaler()
    xembed = scaler.fit_transform(xembed)

    if bundle_dimension > 0 and bundle_dimension < xembed.shape[1]:
        pca = PCA(n_components=bundle_dimension)
        xembed = pca.fit_transform(xembed)

    dmat = compute_sparse_laplace_data(eindex)
    xsvd = np.transpose(
        compute_laplace_eigen_basis(eindex, dmat, embedding_dimension, niter=1000)
    )

    xembed = np.concatenate([xembed], axis=1)

    print('xembed.shape = ' + str(xembed.shape))

    eglist = convert_edge_index_to_list(eindex)
    ylprob = convert_label_to_lprob(ylabel, eps=1.0e-10)

    if dataset_name in ['Chameleon', 'Squirrel']:
        idx_train = data.train_mask[:, split_id].cpu().numpy().nonzero()[0].astype(np.int64)
        idx_valid = data.val_mask[:, split_id].cpu().numpy().nonzero()[0].astype(np.int64)
        idx_ttest = data.test_mask[:, split_id].cpu().numpy().nonzero()[0].astype(np.int64)
    else:
        idx_train = data.train_mask.cpu().numpy().nonzero()[0].astype(np.int64)
        idx_valid = data.val_mask.cpu().numpy().nonzero()[0].astype(np.int64)
        idx_ttest = data.test_mask.cpu().numpy().nonzero()[0].astype(np.int64)

    return xembed, eindex, eglist, ylabel, ylprob, xsvd, idx_train, idx_ttest, idx_valid

def load_batch(qdata, edata, nsample):
    ndata, dimq = qdata.shape
    perm0 = np.random.permutation(ndata)
    perm1 = np.random.permutation(ndata)
    e0 = edata[perm0[:nsample], :]
    q0 = qdata[perm0[:nsample], :]
    e1 = edata[perm1[:nsample], :]
    q1 = qdata[perm1[:nsample], :]
    return q0, e0, q1, e1