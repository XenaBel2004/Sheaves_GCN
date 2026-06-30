import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

def lift(x: torch.Tensor, k: int) -> torch.Tensor:
    return x.repeat(1, 2 * k)


def pool(x: torch.Tensor, k: int) -> torch.Tensor:
    n, d = x.shape
    c = d // (2 * k)
    x = x.view(n, 2 * k, c)
    return x.mean(dim=1)


def to_edge_index(wgraph, device=None):
    if isinstance(wgraph, np.ndarray):
        ei = torch.from_numpy(wgraph)
    else:
        ei = wgraph

    if device is None:
        device = ei.device

    if ei.ndim == 2 and ei.shape[0] == ei.shape[1]:
        src, dst = torch.nonzero(ei, as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0).long().to(device)
        return edge_index

    if ei.ndim == 2 and ei.shape[0] == 2:
        return ei.long().to(device)
    if ei.ndim == 2 and ei.shape[1] == 2:
        return ei.t().contiguous().long().to(device)


def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, rev], dim=1)


def edge_message_passing(x: torch.Tensor,
                         edge_index: torch.Tensor,
                         edge_mlp: nn.Module) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    e = torch.cat([x[src], x[dst]], dim=1)
    m = edge_mlp(e)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, m)
    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    ones = torch.ones(dst.size(0), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1.0)
    out = out / deg.unsqueeze(1)

    return out

class PINN(nn.Module):
    def __init__(
        self,
        matrix_dim: int,
        matrix_type: str,
        hidden_dims: list[int],
        t_domain: tuple[float, float],
        activation: str = "silu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        assert matrix_type in ("full", "compressed")
        input_dim = matrix_dim * matrix_dim if matrix_type == "full" else matrix_dim - 1

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim + 1 + matrix_dim, hidden_dims[0]),
            get_activation(activation),
        )

        hidden_layers = []
        dims = [hidden_dims[0]] + list(hidden_dims)
        for i in range(len(dims) - 1):
            hidden_layers.extend([nn.Linear(dims[i], dims[i + 1]), get_activation(activation)])
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(dims[-1], matrix_dim)

        self.register_buffer("lb", torch.tensor(t_domain[0], requires_grad=False))
        self.register_buffer("ub", torch.tensor(t_domain[1], requires_grad=False))

        self.matrix_dim = matrix_dim
        self.matrix_type = matrix_type
        self.to(dtype)

    def forward(self, t: torch.Tensor, x: torch.Tensor, u0: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        t = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        if self.matrix_type == "full" and x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        h = self.input_layer(torch.column_stack([t, x, u0]))
        h = self.hidden_layers(h)
        ut = self.output_layer(h)
        return {"ut_theta": ut}


class BuNNLayerStrict(nn.Module):
    def __init__(
        self,
        input_dim: int,
        t: float = 1.0,
        K: int = 2,
        matrix_dim: int = 64,
        pinn_ckpt_path: str = "model_best.pth",
        pinn_hidden: tuple[int, int, int] = (512, 512, 512),
        pinn_activation: str = "silu",
        pinn_t_domain: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dimq = int(matrix_dim)
        self.t = float(t)
        self.K = int(K)

        self.proj_a = nn.Linear(input_dim, self.dimq - 1, bias=False)
        self.proj_z = nn.Linear(input_dim, self.dimq, bias=False)
        self.unproj = nn.Linear(self.dimq, input_dim, bias=False)

        self.pinn = PINN(
            matrix_dim=self.dimq,
            matrix_type="compressed",
            hidden_dims=list(pinn_hidden),
            t_domain=pinn_t_domain,
            activation=pinn_activation,
            dtype=torch.float64,
        )

        ckpt = torch.load(pinn_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.pinn.load_state_dict(state_dict)
        self.pinn.eval()
        for p in self.pinn.parameters():
            p.requires_grad = False

        self.double()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, c = X.shape
        if c != self.input_dim:
            raise ValueError(f"BuNNLayerStrict expected dim {self.input_dim}, got {c}")

        a = self.proj_a(X)  # (n, dimq-1)
        z = self.proj_z(X)  # (n, dimq)

        t_batch = torch.full((n, 1), self.t, dtype=X.dtype, device=X.device)
        y = self.pinn(t=t_batch, x=a, u0=z)["ut_theta"]  # (n, dimq)

        out = self.unproj(y)  # (n, input_dim)
        return out

class Sheaf_NNet(nn.Module):
    def __init__(
        self,
        nvert: int,
        dimx: int,
        nlab: int,
        t: float = 1.0,
        k: int = 2,
        K: int = 5,
        nblocks: int = 1,
        make_graph_undirected: bool = True,
        use_residual: bool = True,
        pinn_ckpt_path: str = "model_best.pth",

    ):
        super().__init__()
        self.nvert = int(nvert)
        self.dimx = int(dimx)
        self.nlab = int(nlab)
        self.t = float(t)
        self.k = int(k)
        self.K = int(K)
        self.nblocks = int(nblocks)
        self.make_graph_undirected = bool(make_graph_undirected)
        self.use_residual = bool(use_residual)

        self.lifted_dim = 2 * self.k * self.dimx  # D


        self.edge_mlps = nn.ModuleList()
        self.bunns = nn.ModuleList()
        print("dimx =", self.dimx)
        print("lifted_dim =", self.lifted_dim)

        for _ in range(self.nblocks):
            self.edge_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * self.lifted_dim, self.lifted_dim),
                    nn.ReLU(),
                    nn.Linear(self.lifted_dim, self.lifted_dim),
                ).double()
            )
            self.bunns.append(
                BuNNLayerStrict(
                    input_dim=self.lifted_dim,
                    t=self.t,
                    K=self.K,
                    matrix_dim=64,
                    pinn_ckpt_path=pinn_ckpt_path,
                )
            )

        hidden = 2 * self.dimx
        self.classifier = nn.Sequential(
            nn.Linear(self.dimx, hidden),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden, self.nlab),
            nn.LogSoftmax(dim=1),
        ).double()

    def forward(self, xembed, ylabel, ylprob, wgraph, idvert, node_ids=None):

        xinput = xembed

        xmap = lift(xinput, self.k)  # (N, D)

        edge_index = to_edge_index(wgraph, device=xmap.device)
        if self.make_graph_undirected:
            edge_index = make_undirected(edge_index)

        for edge_mlp, bunn in zip(self.edge_mlps, self.bunns):
            msg = edge_message_passing(xmap, edge_index, edge_mlp)  # (N, D)
            if self.use_residual:
                msg = msg + xmap
            xmap = bunn(msg)
            xmap = F.relu(xmap)

        xmap_pooled = pool(xmap, self.k)  # (N, dimx)

        loss_smap = torch.mean((xmap_pooled - xinput) ** 2) * self.dimx

        if idvert.size > 0:
            glprob = self.classifier(xmap_pooled[idvert])
            kl = torch.sum(torch.exp(ylprob[idvert]) * (ylprob[idvert] - glprob), dim=1)
            loss_lbpr = torch.mean(kl)

            yscore = glprob.cpu().detach().numpy()
            ynumer = yscore.argmax(axis=1)
            loss_accs = accuracy_score(ylabel[idvert], ynumer)
        else:
            loss_lbpr = 0.0 * torch.sum(xinput)
            loss_accs = float("nan")

        return (
            torch.tensor(0.0, dtype=xembed.dtype, device=xembed.device),
            torch.tensor(0.0, dtype=xembed.dtype, device=xembed.device),
            loss_smap,
            loss_lbpr,
            loss_accs,
        )

    def label_inference(self, xembed, wgraph, idx_target, node_ids=None):
        n_local = xembed.shape[0]

        if node_ids is None:
            node_ids = torch.arange(n_local, device=xembed.device, dtype=torch.long)
        else:
            node_ids = torch.as_tensor(node_ids, device=xembed.device, dtype=torch.long)
        xinput = xembed

        xmap = lift(xinput, self.k)
        edge_index = to_edge_index(wgraph, device=xmap.device)
        if self.make_graph_undirected:
            edge_index = make_undirected(edge_index)

        for edge_mlp, bunn in zip(self.edge_mlps, self.bunns):
            msg = edge_message_passing(xmap, edge_index, edge_mlp)
            if self.use_residual:
                msg = msg + xmap
            xmap = F.relu(bunn(msg))

        xmap_pooled = pool(xmap, self.k)
        glprob = self.classifier(xmap_pooled)
        return glprob[idx_target].argmax().item()

    def infer_all_labels(self, xembed, wgraph):
        xinput = xembed

        xmap = lift(xinput, self.k)
        edge_index = to_edge_index(wgraph, device=xmap.device)
        if self.make_graph_undirected:
            edge_index = make_undirected(edge_index)

        for edge_mlp, bunn in zip(self.edge_mlps, self.bunns):
            msg = edge_message_passing(xmap, edge_index, edge_mlp)
            if self.use_residual:
                msg = msg + xmap
            xmap = F.relu(bunn(msg))

        xmap_pooled = pool(xmap, self.k)
        glprob = self.classifier(xmap_pooled)
        return glprob.argmax(dim=1)