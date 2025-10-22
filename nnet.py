import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from nnet_PINN import Mat_Exp_NNet

def lift(x, k):
    n, c = x.shape
    lifted = torch.zeros(n, 2 * c * k, device=x.device, dtype=x.dtype)
    for i in range(k):
        lifted[:, 2 * i * c: 2 * i * c + c] = x
    return lifted

def pool(x, k):
    n, d = x.shape
    c = d // (2 * k)
    pooled = torch.zeros(n, c, device=x.device, dtype=x.dtype)
    for i in range(k):
        pooled += x[:, 2 * i * c: 2 * i * c + c]
    return pooled


class Orthogonal2D(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.angles = nn.Parameter(torch.randn(num_blocks))
        self.use_reflection = torch.arange(num_blocks) % 2 == 1

    def forward(self):
        theta = self.angles
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        r = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)
        r_star = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([sin, -cos], dim=-1)
        ], dim=-2)

        return torch.where(self.use_reflection[:, None, None], r_star, r)


class BuNNLayerStrict(nn.Module):
    def __init__(self, input_dim, t=1.0, K=2):

        super().__init__()
        self.num_blocks = input_dim // 2
        self.t = t
        self.K = K
        self.O = Orthogonal2D(self.num_blocks)
        self.W = nn.Parameter(torch.eye(2).repeat(self.num_blocks, 1, 1))
        self.b = nn.Parameter(torch.zeros(self.num_blocks, 2))
        dimq = 20
        fname = './mat_exp_net.ptr'
        self.mat_exp_net = Mat_Exp_NNet(dimq)
        self.mat_exp_net.load_state_dict(torch.load(fname, map_location='cpu'))
        self.mat_exp_net.eval()
        for p in self.mat_exp_net.parameters():
            p.requires_grad = False

    def forward(self, X):
        n, c = X.shape
        b = self.num_blocks

        X_b = X.view(n, b, 2)
        O = self.O()
        X_rot = torch.einsum("bij,nbi->nbj", O, X_b)
        H = torch.einsum("bij,nbj->nbi", self.W, X_rot) + self.b  # (n, b, 2)

        H_flat = H.view(n, -1)

        q = torch.zeros((n, H_flat.shape[1] + 1), dtype=torch.double, device=H.device)
        q[:, 0] = 1.0
        q[:, 1:] = H_flat

        smat = self.mat_exp_net.s_matrix(q)

        out = smat.view(n, c)
        return out


class Sheaf_NNet(nn.Module):
    def __init__(self, nvert, dimx, nlab, nconv=3, t=1.0, k=2, K=5):
        super().__init__()
        self.nvert = nvert
        self.dimx = dimx
        self.nlab = nlab
        self.t = t
        self.k = k
        self.K = K
        self.lifted_dim = 2 * k * dimx

        self.bunn_layers = nn.ModuleList()
        for i in range(nconv):
            in_dim = self.lifted_dim
            self.bunn_layers.append(BuNNLayerStrict(in_dim, t, K))

        self.classifier = nn.Sequential(
            nn.Linear(self.dimx, self.nlab),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xembed, ylabel, ylprob, wgraph, idvert):

        L = wgraph
        xproj = lift(xembed, self.k)
        xmaped = xproj
        for layer in self.bunn_layers:
            xmaped = F.relu(layer(xmaped))
        xmaped_pooled = pool(xmaped, self.k)
        loss_smap = torch.mean((xmaped_pooled - xembed)**2) * self.dimx
        if idvert.size > 0:
            glprob = self.classifier(xmaped_pooled[idvert])
            kl_div = torch.sum(torch.exp(ylprob[idvert]) * (ylprob[idvert] - glprob), dim=1)
            loss_lbpr = torch.mean(kl_div)
        else:
            loss_lbpr = 0.0 * torch.sum(xembed)

        yscore = self.classifier(xmaped_pooled[idvert]).cpu().detach().numpy()
        ynumer = yscore.argmax(axis=1)
        loss_accs = accuracy_score(ylabel[idvert], ynumer)

        return (torch.tensor(0.0), torch.tensor(0.0), loss_smap, loss_lbpr, loss_accs)

    def label_inference(self, xembed, wgraph, idx_target):
        xproj = lift(xembed, self.k)
        xmaped = xproj
        for layer in self.bunn_layers:
            xmaped = F.relu(layer(xmaped))
        xmaped_pooled = pool(xmaped, self.k)
        glprob = self.classifier(xmaped_pooled)
        return glprob[idx_target].argmax().item()
