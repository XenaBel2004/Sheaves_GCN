import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
from nnet_PINN import Mat_Exp_NNet

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(nbatch, dimq, scaling_factor=1.0):
    t = scaling_factor * 2.0 * (np.random.rand(nbatch) - 0.5)

    omega = scaling_factor * np.random.normal(0.0, 1.0, (nbatch, dimq, dimq))
    for idx in range(nbatch):
        omega[idx, :, :] = 0.5 * (np.transpose(omega[idx, :, :]) - omega[idx, :, :])

    q_i = np.zeros((nbatch, dimq * dimq + 1))
    q_i[:, 1:] = omega.reshape(nbatch, dimq * dimq)[:, :]
    q_e = np.zeros((nbatch, dimq * dimq + 1))
    q_e[:, 1:] = omega.reshape(nbatch, dimq * dimq)[:, :]

    for idx in range(nbatch):
        q_e[idx, :] = q_e[idx, :] + t[idx]
    return (torch.from_numpy(q_i).double(), torch.from_numpy(q_e).double())

def compute_nnet_params(mat_exp_net, nepoch, nbatch, lr=1.0e-3):
    mat_exp_net.s_matrix.requires_grad_(True)
    optimizer = optim.Adam(mat_exp_net.parameters(), lr=lr, weight_decay=0.0)
    for kepoch in range(nepoch):
        print('epoch ' + str(kepoch) + ' of ' + str(nepoch))
        scaling_factor = min((2.0 + (1 + kepoch) / nepoch) ** 12.0, 1.0)
        q_i, q_e = generate_data(nbatch, mat_exp_net.dimq, scaling_factor=scaling_factor)
        loss = mat_exp_net.forward(Variable(q_i), Variable(q_e))
        loss.backward()
        optimizer.step()
    return 0

def compute_matrix_exponent(nnet, qmat):
    q = np.zeros((qmat.size + 1))
    q[0] = 1.0
    q[1:] = qmat.flatten()
    smat = np.reshape(nnet.s_matrix(torch.from_numpy(q.reshape(1, -1)).double()).cpu().detach().numpy(), qmat.shape)
    return smat

def main():
    print('inside the main function')
    dimq = 50
    mat_exp_net = Mat_Exp_NNet(dimq)
    nepoch = 10000
    nbatch = 16
    lr = 1.0e-5
    compute_nnet_params(mat_exp_net, nepoch, nbatch, lr=lr)

    # сохранить только веса
    fname = './nnet_folder/mat_exp_net.ptr'
    torch.save(mat_exp_net.state_dict(), fname)

    # simple test
    qmat = np.random.normal(0.0, 1.0, (dimq, dimq))
    qmat = 0.5 * qmat - 0.5 * np.transpose(qmat) #делаем кососимм

    # создать новую модель и загрузить веса
    mat_exp_net = Mat_Exp_NNet(dimq)
    mat_exp_net.load_state_dict(torch.load(fname, map_location='cpu'))
    mat_exp_net.eval()  # инференс

    smat = compute_matrix_exponent(mat_exp_net, qmat)
    print(smat)
    print(np.dot(np.transpose(smat), smat))
    res = np.max(np.absolute(smat - np.eye(dimq)))
    print('res = ' + str(res))
    return 0

main()
