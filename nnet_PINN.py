import numpy as np
import torch
from torch import nn

class Mat_Exp_NNet(nn.Module):
    def __init__(self, dimq, loss_target=0.01, n0=120, n1=600):
        super(Mat_Exp_NNet, self).__init__()
        self.dimq = dimq
        self.loss_target = loss_target
        self.s_matrix = nn.Sequential(nn.Linear(self.dimq * self.dimq + 1, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n0),
                                                nn.ReLU(),
                                                nn.Linear(n0, n1),
                                                nn.ReLU(),                                                   
                                                nn.Linear(n1, self.dimq * self.dimq))
        self.double()

    def initial_conditions(self, nbatch):
        s0 = np.eye(self.dimq, self.dimq)
        smat_0 = np.zeros((nbatch, self.dimq, self.dimq))
        for idx in range(nbatch):
            smat_0[idx, :, :] = s0[:, :]
        return torch.from_numpy(smat_0).double()

    def gradient_exact(self, q):
        q = q.clone().detach().requires_grad_(True)
        y = self.s_matrix(q)
        grads = []
        for i in range(y.shape[1]):
            grad_i = torch.autograd.grad(
                outputs=y[:, i],
                inputs=q,
                grad_outputs=torch.ones_like(y[:, i]),
                retain_graph=True,
                create_graph=True)[0][:, 0]
            grads.append(grad_i)
        dt_deriv = torch.stack(grads, dim=1)
        return dt_deriv.reshape(-1, self.dimq, self.dimq).double()

    def compute_loss_weight_simple(self, loss0):
        l0 = loss0.cpu().detach().numpy() / self.loss_target

        w0 = 1.0
        w1 = np.exp(-l0)

        w_summ = w0 + w1
        w0 = w0 / w_summ
        w1 = w1 / w_summ
        return (w0, w1)




    def forward(self, q_i, q_e, eps=1.0e-6):
        ### in principle< you have all the data in dimensions
        smat_i = torch.reshape(self.s_matrix(q_i), (-1, self.dimq, self.dimq))
        smat_0 = self.initial_conditions(q_i.shape[0])
        loss_i = torch.mean((smat_i - smat_0) * (smat_i - smat_0))
        smat_e = torch.reshape(self.s_matrix(q_e), (-1, self.dimq, self.dimq))
        omega = torch.reshape(q_e[:, 1:], (-1, self.dimq, self.dimq))
        dsdt = torch.reshape(self.gradient_exact(q_e), (-1, self.dimq, self.dimq))

        ### now you can compute the resdisual
        diff = dsdt - torch.bmm(omega, smat_e)
        loss_e = torch.mean(diff * diff)

        w_i, w_e, = self.compute_loss_weight_simple(loss_i)
        loss = w_i * loss_i + w_e * loss_e
        print('w_i = ' + str(w_i))
        print('w_e = ' + str(w_e))
        print('loss_i = ' + str(loss_i.item()))
        print('loss_e = ' + str(loss_e.item()))
        print('loss = ' + str(loss.item()))
        return loss