import numpy as np
from numba import jit
import torch

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math_support import compute_qr_factorization
from nnet import Sheaf_NNet
from sheaf_calculator import compute_neural_network_parameters
from data_loader import read_data, perform_ttv_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import ndcg_score



def data_preprocessing(user_embed, item_embed, dimy):
    nuser, dime = user_embed.shape
    nitem, dime = item_embed.shape
    print('nuser = ' + str(nuser))
    print('nitem = ' + str(nitem))
    xembed = np.zeros((nuser + nitem, dimy))
    xembed[:nuser, :] = user_embed[:, :min(dime, dimy)]
    xembed[nuser:, :] = item_embed[:, :min(dime, dimy)]
    np.save('./data/xembed.npy', xembed)
    return xembed


def learn_sheaf_parameters(xembed, ylabel, ylprob, eglist,
                           idx_train, idx_ttest, idx_valid,
                           dimy=16, nbatch=128, nepoch=2048,
                           lr=1.0e-3, unique_name=''):
    print('learning_the_sheaf')
    nvert, dimx = xembed.shape
    nlab = np.int64(np.max(ylabel) + 1)
    nnet = Sheaf_NNet(nvert, dimx, dimy, nlab).double()

   

    ltrain, lttest, lvalid, accs_train, accs_ttest, accs_valid = compute_neural_network_parameters(nnet, nepoch, nbatch, lr,
                                                                                                   xembed, ylabel, ylprob, eglist,
                                                                                                   idx_train, idx_ttest, idx_valid)


    print('accs_train = ' + str(accs_train))
    print('accs_ttest = ' + str(accs_ttest))
    print('accs_valid = ' + str(accs_valid))
    nnet_folder = './nnet_folder'
    fname = nnet_folder + '/nnet_sheaf' + unique_name + '.ptr'
    torch.save(nnet, fname)    
    np.save('./results/loss_data_train' + unique_name + '.npy', ltrain)
    np.save('./results/loss_data_ttest' + unique_name + '.npy', lttest)
    np.save('./results/loss_data_valid' + unique_name + '.npy', lvalid)
    np.save('./results/accs_train' + unique_name + '.npy', accs_train)
    np.save('./results/accs_ttest' + unique_name + '.npy', accs_ttest)
    np.save('./results/accs_valid' + unique_name + '.npy', accs_valid)

    return 0



def data_postprocessing(dime, dimy, conv_depth):
    unique_name = unique_name = '_dime_' + str(dime) + '_dimy_' + str(dimy)
    loss_data_train = np.load('./results/loss_data_train' + unique_name + '.npy')
    loss_data_ttest = np.load('./results/loss_data_ttest' + unique_name + '.npy')
    loss_data_valid = np.load('./results/loss_data_valid' + unique_name + '.npy')
    numt = loss_data_train.shape[0]
    t = np.linspace(1, numt, numt, dtype=np.int64)
    plt.figure(figsize=(6, 4))
    plt.scatter(t, loss_data_train[:, 0], c='b', label='Train', marker='.')
    plt.scatter(t, loss_data_ttest[:, 0], c='r', label='Test', marker='P')
    plt.scatter(t, loss_data_valid[:, 0], c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'Iteration Number')
    plt.ylabel(r'Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/loss_orth' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_train[:, 0]) / np.log(2.0), c='b', label='Train', marker='.')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_ttest[:, 0]) / np.log(2.0), c='r', label='Test', marker='P')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_valid[:, 0]) / np.log(2.0), c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'$\log_2(Iteration)$')
    plt.ylabel(r'$\log_@(Loss)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/log_log_loss_orth' + unique_name + '.pdf')
    plt.show()


    plt.figure(figsize=(6, 4))
    plt.scatter(t, loss_data_train[:, 1], c='b', label='Train', marker='.')
    plt.scatter(t, loss_data_ttest[:, 1], c='r', label='Test', marker='P')
    plt.scatter(t, loss_data_valid[:, 1], c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'Iteration Number')
    plt.ylabel(r'Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/loss_cons' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_train[:, 1]) / np.log(2.0), c='b', label='Train', marker='.')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_ttest[:, 1]) / np.log(2.0), c='r', label='Test', marker='P')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_valid[:, 1]) / np.log(2.0), c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'$\log_2(Iteration)$')
    plt.ylabel(r'$\log_@(Loss)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/log_log_loss_cons' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(t, loss_data_train[:, 2], c='b', label='Train', marker='.')
    plt.scatter(t, loss_data_ttest[:, 2], c='r', label='Test', marker='P')
    plt.scatter(t, loss_data_valid[:, 2], c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'Iteration Number')
    plt.ylabel(r'Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/loss_smap' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_train[:, 2]) / np.log(2.0), c='b', label='Train', marker='.')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_ttest[:, 2]) / np.log(2.0), c='r', label='Test', marker='P')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_valid[:, 2]) / np.log(2.0), c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'$\log_2(Iteration)$')
    plt.ylabel(r'$\log_@(Loss)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/log_log_loss_smap' + unique_name + '.pdf')
    plt.show()


    plt.figure(figsize=(6, 4))
    plt.scatter(t, loss_data_train[:, 3], c='b', label='Train', marker='.')
    plt.scatter(t, loss_data_ttest[:, 3], c='r', label='Test', marker='P')
    plt.scatter(t, loss_data_valid[:, 3], c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'Iteration Number')
    plt.ylabel(r'Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/loss_lmse' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_train[:, 3]) / np.log(2.0), c='b', label='Train', marker='.')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_ttest[:, 3]) / np.log(2.0), c='r', label='Test', marker='P')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_valid[:, 3]) / np.log(2.0), c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'$\log_2(Iteration)$')
    plt.ylabel(r'$\log_@(Loss)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/log_log_loss_lmse' + unique_name + '.pdf')
    plt.show()


    plt.figure(figsize=(6, 4))
    plt.scatter(t, loss_data_train[:, 4], c='b', label='Train', marker='.')
    plt.scatter(t, loss_data_ttest[:, 4], c='r', label='Test', marker='P')
    plt.scatter(t, loss_data_valid[:, 4], c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'Iteration Number')
    plt.ylabel(r'Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/loss_comb' + unique_name + '.pdf')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_train[:, 4]) / np.log(2.0), c='b', label='Train', marker='.')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_ttest[:, 4]) / np.log(2.0), c='r', label='Test', marker='P')
    plt.scatter(np.log(t) / np.log(2.0), np.log(loss_data_valid[:, 4]) / np.log(2.0), c='g', label='Validation', marker='s')
    plt.grid()
    plt.xlabel(r'$\log_2(Iteration)$')
    plt.ylabel(r'$\log_@(Loss)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./post_processing/log_log_loss_comb' + unique_name + '.pdf')
    plt.show()


    nnet_folder = './nnet_folder'
    fname = nnet_folder + '/nnet_sheaf' + unique_name + '.ptr'
    nnet = torch.load(fname)
    print(nnet)
    xembed = np.load('./data/xembed' + unique_name + '.npy')
    wtrain = np.load('./data/wtrain' + unique_name + '.npy')
    wttest = np.load('./data/wttest' + unique_name + '.npy')
    wvalid = np.load('./data/wvalid' + unique_name + '.npy')
    n_user = compute_number_of_users(wtrain)
    xsheaf = nnet.compute_graph_features(torch.from_numpy(0.0 + xembed), 
                                         torch.from_numpy(0.0 + wtrain),
                                         conv_depth)
    print('xsheaf.shape = ' + str(xsheaf.shape))
    x_diff = np.max(np.absolute(xsheaf - xembed))
    print('x_diff = ' + str(x_diff))

    user_embed = xembed[:n_user, :]
    item_embed = xembed[n_user:, :]
    user_sheaf = xsheaf[:n_user, :]
    item_sheaf = xsheaf[n_user:, :]

    ctrain = recover_original_matrix(wtrain)
    cttest = recover_original_matrix(wttest)
    cvalid = recover_original_matrix(wvalid)

    rmse_embed_train = compute_rmse(user_embed, item_embed, ctrain)
    rmse_embed_ttest = compute_rmse(user_embed, item_embed, cttest)
    rmse_embed_valid = compute_rmse(user_embed, item_embed, cvalid)

    rmse_sheaf_train = compute_rmse(user_sheaf, item_sheaf, ctrain)
    rmse_sheaf_ttest = compute_rmse(user_sheaf, item_sheaf, cttest)
    rmse_sheaf_valid = compute_rmse(user_sheaf, item_sheaf, cvalid)

    print('rmse_embed_train = ' + str(rmse_embed_train))
    print('rmse_embed_ttest = ' + str(rmse_embed_ttest))
    print('rmse_embed_valid = ' + str(rmse_embed_valid))

    print('rmse_sheaf_train = ' + str(rmse_sheaf_train))
    print('rmse_sheaf_ttest = ' + str(rmse_sheaf_ttest))
    print('rmse_sheaf_valid = ' + str(rmse_sheaf_valid))

    cembed = compute_labels(user_embed, item_embed, threshold=0.5)
    csheaf = compute_labels(user_embed, item_embed, threshold=0.5)
    accs_embed_train = accuracy_score(ctrain.flatten(), cembed.flatten())
    accs_embed_ttest = accuracy_score(cttest.flatten(), cembed.flatten())
    accs_embed_valid = accuracy_score(cvalid.flatten(), cembed.flatten())
    accs_sheaf_train = accuracy_score(ctrain.flatten(), csheaf.flatten())
    accs_sheaf_ttest = accuracy_score(cttest.flatten(), csheaf.flatten())
    accs_sheaf_valid = accuracy_score(cvalid.flatten(), csheaf.flatten())

    print('accs_embed_train = ' + str(accs_embed_train))
    print('accs_embed_ttest = ' + str(accs_embed_ttest))
    print('accs_embed_valid = ' + str(accs_embed_valid))

    print('accs_sheaf_train = ' + str(accs_sheaf_train))
    print('accs_sheaf_ttest = ' + str(accs_sheaf_ttest))
    print('accs_sheaf_valid = ' + str(accs_sheaf_valid))

    return 0


def main():
    print('inside the main function')
    embedding_dimension = 256
    sheaf_edge_space_dimension = 64
    dataset_name = 'CiteSeer'
    # dataset_name = 'PubMed'
    # dataset_name = 'Cora'
    # dataset_name = 'DBLP'
    unique_name_base = '_svd_dime_' + str(embedding_dimension) + '_dimy_' + str(sheaf_edge_space_dimension)
    first_time_flag = True

    nreal = 10
    for ireal in range(nreal):
        unique_name = '_' + dataset_name + '_realization_' + str(ireal) + unique_name_base
        if first_time_flag:
            xembed, eglist, ylabel, ylprob, xsvd = read_data(embedding_dimension=embedding_dimension,
                                                             dataset_name=dataset_name, eps=1.0e-6)
            ### you have to modify read_data method
            print('ylabel.shape = ' + str(ylabel.shape))
            nsample = xembed.shape[0]
            idx_train, idx_ttest, idx_valid = perform_ttv_split(nsample, ftrain=0.8, fttest=0.1, fvalid=0.1)
            # np.save('./data/idx_train' + unique_name + '.npy', idx_train)
            # np.save('./data/idx_ttest' + unique_name + '.npy', idx_ttest)
            # np.save('./data/idx_valid' + unique_name + '.npy', idx_valid)
            # np.save('./data/xembed' + unique_name + '.npy', xembed)
            # np.save('./data/xsvd' + unique_name + '.npy', xembed)
            # np.save('./data/wgraph' + unique_name + '.npy', wgraph)

        else:
            pass
            # eglist = np.load('./data/wgraph' + unique_name + '.npy')
            # xembed = np.load('./data/xembed' + unique_name + '.npy')
            # idx_train = np.load('./data/idx_train' + unique_name + '.npy')
            # idx_ttest = np.load('./data/idx_ttest' + unique_name + '.npy')
            # idx_valid = np.load('./data/idx_valid' + unique_name + '.npy')

        print('xembed.shape = ' + str(xembed.shape))
        print(unique_name)
        learn_sheaf_parameters(xsvd, ylabel, ylprob, eglist,
                               idx_train, idx_ttest, idx_valid,
                               dimy=sheaf_edge_space_dimension,
                               nbatch=512, nepoch=(64 * 1024),
                               lr=1.0e-3, unique_name=unique_name)
    return 0

main()









