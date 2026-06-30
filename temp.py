import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(
        self,
        n: int,
        k: int,
        t_domain: tuple[float, float],
        hidden_dim: int,
        num_xu_blocks: int,
        num_fusion_blocks: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n = n

        t_input_dim = 1

        self.t_encoder = nn.Sequential(
            nn.Linear(t_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        xu_layers = []
        in_ch = k + 1
        for _ in range(num_xu_blocks):
            xu_layers.extend(
                [
                    nn.Conv1d(in_ch, hidden_dim, kernel_size, padding='same'),
                    nn.SiLU(),
                ]
            )
            in_ch = hidden_dim
        self.xu_encoder = nn.Sequential(*xu_layers)

        fusion_layers = []
        in_ch = hidden_dim + hidden_dim
        for _ in range(num_fusion_blocks):
            fusion_layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size, padding='same'),
                nn.SiLU(),
            ]
            in_ch = hidden_dim
        self.fusion = nn.Sequential(*fusion_layers)
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.register_buffer('t_min', torch.tensor(t_domain[0], dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_domain[1], dtype=torch.float32))

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, u0: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        x_matrix = torch.cat([x, u0.unsqueeze(1)], dim=1)

        t_h = self.t_encoder(t_norm)
        t_h_exp = t_h.unsqueeze(-1).expand(-1, -1, self.n)
        xu_h = self.xu_encoder(x_matrix)
        h = torch.cat([xu_h, t_h_exp], dim=1)
        h = self.fusion(h)
        ut = self.output_proj(h).squeeze(1)
        return {'ut': ut}


N = 128
K = 1
T_DOMAIN = (0.0, 1.0)
hidden_dim = 32
num_xu_blocks = 1
num_fusion_blocks = 3
kernel_size = 3

model = PINN(
    n=N,
    k=K,
    t_domain=T_DOMAIN,
    hidden_dim=hidden_dim,
    num_xu_blocks=num_xu_blocks,
    num_fusion_blocks=num_fusion_blocks,
    kernel_size=kernel_size,
)

checkpoint = torch.load(
    '/Users/tlidzhiev/Desktop/projects/exp-matrix-pinn/n-64-k-1-model-best-1250000.pth',
    map_location='cpu',
    weights_only=False,
)

raw = checkpoint['ema']['shadow']
state_dict = {}
for key, val in raw.items():
    if key.startswith('_orig_mod.solver.'):
        state_dict[key[len('_orig_mod.solver.') :]] = val
    elif key.startswith('_orig_mod.'):
        state_dict[key[len('_orig_mod.') :]] = val
    else:
        state_dict[key] = val

model.load_state_dict(state_dict)
model.eval()

print(model)


# --- metrics as plain functions ---


def relative_l2_error(
    ut_pred: torch.Tensor, ut: torch.Tensor, u0: torch.Tensor, eps: float = 1e-8
) -> float:
    # ut_pred, ut: (B, T, n); u0: (B, n)
    diff_norm = torch.linalg.vector_norm(ut - ut_pred, dim=-1)  # (B, T)
    u0_norm = torch.linalg.vector_norm(u0, dim=-1, keepdim=True)  # (B, 1)
    return (diff_norm / (u0_norm + eps)).mean().item()


def cosine_similarity(ut_pred: torch.Tensor, ut: torch.Tensor) -> float:
    # ut_pred, ut: (B, T, n)
    import torch.nn.functional as F

    return F.cosine_similarity(ut, ut_pred, dim=-1).mean().item()


def relative_norm_drift(ut_pred: torch.Tensor, u0: torch.Tensor, eps: float = 1e-8) -> float:
    # ut_pred: (B, T, n); u0: (B, n)
    pred_norm = torch.linalg.vector_norm(ut_pred, dim=-1)  # (B, T)
    u0_norm = torch.linalg.vector_norm(u0, dim=-1, keepdim=True)  # (B, 1)
    return (torch.abs(pred_norm - u0_norm) / (u0_norm + eps)).mean().item()


# --- sample a small batch ---


def sample_batch(
    n: int,
    k: int,
    batch_size: int,
    num_time_points: int,
    t_domain: tuple,
    trunc_bounds: tuple,
):
    rng = torch.Generator()
    t_min, t_max = t_domain
    trunc_lower, trunc_upper = trunc_bounds
    t = torch.linspace(t_min, t_max, num_time_points)  # (T,)

    rows, cols = torch.triu_indices(n, n, offset=1)
    mask = (cols - rows) <= k
    rows, cols = rows[mask], cols[mask]
    num_vals = rows.shape[0]

    vals = torch.nn.init.trunc_normal_(
        torch.empty(batch_size, num_vals), a=trunc_lower, b=trunc_upper, generator=rng
    )
    u0 = torch.nn.init.trunc_normal_(
        torch.empty(batch_size, n), a=trunc_lower, b=trunc_upper, generator=rng
    )

    # build dense skew-symmetric X for each sample and solve ODE via matrix exp
    X = torch.zeros(batch_size, n, n)
    X[:, rows, cols] = vals
    X[:, cols, rows] = -vals
    # -t: (T,1,1,1)  X: (1,B,n,n)  → tX: (T,B,n,n)  u0: (1,B,n,1) → ut: (T,B,n,1)
    tX = -t.reshape(-1, 1, 1, 1) * X.unsqueeze(0)
    ut = torch.linalg.matrix_exp(tX) @ u0.reshape(1, batch_size, n, 1)
    ut = ut.squeeze(-1).permute(1, 0, 2)  # (B, T, n)

    # sparse diagonal representation: (B, k, n)
    x = torch.zeros(batch_size, k, n)
    diag_idx = cols - rows - 1
    pos_idx = rows
    x[:, diag_idx, pos_idx] = vals

    return {'t': t, 'x': x, 'u0': u0, 'ut': ut}


batch = sample_batch(
    N, K, batch_size=64, num_time_points=100, t_domain=T_DOMAIN, trunc_bounds=(-2.0, 2.0)
)
t, x, u0, ut = batch['t'], batch['x'], batch['u0'], batch['ut']
print('t', t.shape, 'x', x.shape, 'u0', u0.shape, 'ut', ut.shape)

B, num_t, n = ut.shape
with torch.no_grad():
    t_flat = t.unsqueeze(0).expand(B, num_t).reshape(B * num_t, 1)
    x_flat = x.unsqueeze(1).expand(B, num_t, -1, -1).reshape(B * num_t, *x.shape[1:])
    u0_flat = u0.unsqueeze(1).expand(B, num_t, -1).reshape(B * num_t, -1)
    ut_pred = model(t=t_flat, x=x_flat, u0=u0_flat)['ut'].reshape(B, num_t, n)

print(f'relative_l2_error:   {relative_l2_error(ut_pred, ut, u0):.4f}')
print(f'cosine_similarity:   {cosine_similarity(ut_pred, ut):.4f}')
print(f'relative_norm_drift: {relative_norm_drift(ut_pred, u0):.4f}')


num_samples_plot = 4  # samples to show
num_coords_plot = 8  # coordinates per sample

t_np = t.numpy()
gt_np = ut.detach().numpy()  # (B, T, n)
pr_np = ut_pred.detach().numpy()  # (B, T, n)

fig, axes = plt.subplots(
    num_samples_plot,
    num_coords_plot,
    figsize=(num_coords_plot * 2.5, num_samples_plot * 2),
    sharex=True,
)

for i in range(num_samples_plot):
    for j in range(num_coords_plot):
        ax = axes[i, j]
        ax.plot(t_np, gt_np[i, :, j], label='GT', color='steelblue', linewidth=1.5)
        ax.plot(t_np, pr_np[i, :, j], label='Pred', color='tomato', linewidth=1.5, linestyle='--')
        ax.set_title(f's{i} c{j}', fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0 and j == 0:
            ax.legend(fontsize=6)

fig.supxlabel('t')
fig.supylabel('u(t)')
plt.tight_layout()
plt.savefig('trajectories.png', dpi=150)
print('Saved trajectories.png')
