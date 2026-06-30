import os
import ssl
import random
import urllib.request
from collections import defaultdict

import certifi
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_file(url: str, dst: str):
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as response, open(dst, "wb") as f:
        f.write(response.read())


def download_builtin_dataset(dataset_name: str, root_dir: str = "./data"):
    dataset_name = dataset_name.lower()
    dataset_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    url_map = {
        "gowalla": {
            "train.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/train.txt",
            "test.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/test.txt",
        },
        "yelp2018": {
            "train.txt": "https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/master/Data/yelp2018/train.txt",
            "test.txt": "https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/master/Data/yelp2018/test.txt",
        },
    }

    if dataset_name not in url_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for filename, url in url_map[dataset_name].items():
        dst = os.path.join(dataset_dir, filename)
        if not os.path.exists(dst):
            print(f"Downloading {dataset_name}/{filename}...")
            download_file(url, dst)

    return dataset_dir


def read_interaction_file(path: str) -> np.ndarray:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            user = int(parts[0])
            items = [int(x) for x in parts[1:]]
            for item in items:
                pairs.append((user, item))
    if len(pairs) == 0:
        raise ValueError(f"No interactions found in {path}")
    return np.asarray(pairs, dtype=np.int64)


def reindex_pairs(train_pairs: np.ndarray, test_pairs: np.ndarray):
    user_ids = np.unique(np.concatenate([train_pairs[:, 0], test_pairs[:, 0]]))
    item_ids = np.unique(np.concatenate([train_pairs[:, 1], test_pairs[:, 1]]))

    user2id = {u: i for i, u in enumerate(user_ids.tolist())}
    item2id = {it: i for i, it in enumerate(item_ids.tolist())}

    train_re = np.zeros_like(train_pairs)
    test_re = np.zeros_like(test_pairs)

    train_re[:, 0] = [user2id[u] for u in train_pairs[:, 0]]
    train_re[:, 1] = [item2id[it] for it in train_pairs[:, 1]]
    test_re[:, 0] = [user2id[u] for u in test_pairs[:, 0]]
    test_re[:, 1] = [item2id[it] for it in test_pairs[:, 1]]

    return train_re, test_re, user2id, item2id


def build_user_pos_items(pairs: np.ndarray):
    out = defaultdict(list)
    for u, i in pairs:
        out[int(u)].append(int(i))
    return dict(out)


def build_seen_items(pairs: np.ndarray):
    out = defaultdict(set)
    for u, i in pairs:
        out[int(u)].add(int(i))
    return dict(out)


def build_bipartite_edge_index(user_item_pairs: np.ndarray, n_users: int) -> torch.Tensor:
    users = torch.as_tensor(user_item_pairs[:, 0], dtype=torch.long)
    items = torch.as_tensor(user_item_pairs[:, 1], dtype=torch.long) + n_users
    return torch.stack([users, items], dim=0)


def make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    rev = torch.stack([dst, src], dim=0)
    return torch.cat([edge_index, rev], dim=1)


def load_gowalla_or_yelp2018(data_dir: str):
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_pairs_raw = read_interaction_file(train_path)
    test_pairs_raw = read_interaction_file(test_path)

    train_pairs, test_pairs, user2id, item2id = reindex_pairs(train_pairs_raw, test_pairs_raw)

    n_users = len(user2id)
    n_items = len(item2id)

    train_user_pos = build_user_pos_items(train_pairs)
    test_user_pos = build_user_pos_items(test_pairs)
    train_seen = build_seen_items(train_pairs)
    edge_index = build_bipartite_edge_index(train_pairs, n_users)

    overlap = 0
    for u in test_user_pos:
        overlap += len(set(train_user_pos.get(u, [])) & set(test_user_pos.get(u, [])))

    print("n_users =", n_users)
    print("n_items =", n_items)
    print("train_pairs =", len(train_pairs))
    print("test_pairs =", len(test_pairs))
    print("users_with_train =", len(train_user_pos))
    print("users_with_test =", len(test_user_pos))
    print("train_test_overlap =", overlap)

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "n_users": n_users,
        "n_items": n_items,
        "edge_index": edge_index,
        "train_user_pos": train_user_pos,
        "test_user_pos": test_user_pos,
        "train_seen": train_seen,
    }


def sample_bpr_batch(user_pos_items, n_items: int, batch_size: int, device: torch.device):
    users = np.random.choice(list(user_pos_items.keys()), size=batch_size, replace=True)
    pos_items = []
    neg_items = []

    for u in users:
        pos = np.random.choice(user_pos_items[u])
        user_pos_set = set(user_pos_items[u])

        neg = np.random.randint(0, n_items)
        while neg in user_pos_set:
            neg = np.random.randint(0, n_items)

        pos_items.append(pos)
        neg_items.append(neg)

    users = torch.as_tensor(users, dtype=torch.long, device=device)
    pos_items = torch.as_tensor(pos_items, dtype=torch.long, device=device)
    neg_items = torch.as_tensor(neg_items, dtype=torch.long, device=device)
    return users, pos_items, neg_items


def recall_ndcg_at_k(topk_items, gt_items, k: int):
    hits = 0
    dcg = 0.0
    for rank, item in enumerate(topk_items[:k], start=1):
        if item in gt_items:
            hits += 1
            dcg += 1.0 / np.log2(rank + 1)

    recall = hits / max(1, len(gt_items))
    ideal_hits = min(len(gt_items), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg


def lightgcn_propagate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]

    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    deg.index_add_(0, src, torch.ones(src.size(0), device=x.device, dtype=x.dtype))
    deg = deg.clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)

    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    out = torch.zeros_like(x)
    weighted_messages = x[src] * norm.unsqueeze(1)
    out.index_add_(0, dst, weighted_messages)
    return out


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


class PINN(nn.Module):
    def __init__(
        self,
        matrix_dim: int,
        matrix_type: str,
        hidden_dims: list[int],
        t_domain: tuple[float, float],
        activation: str = "silu",
        dtype: torch.dtype = torch.float32,
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
            hidden_layers.extend(
                [nn.Linear(dims[i], dims[i + 1]), get_activation(activation)]
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(dims[-1], matrix_dim)

        self.register_buffer("lb", torch.tensor(t_domain[0], requires_grad=False, dtype=dtype))
        self.register_buffer("ub", torch.tensor(t_domain[1], requires_grad=False, dtype=dtype))

        self.matrix_dim = matrix_dim
        self.matrix_type = matrix_type
        self.to(dtype)

    def forward(self, t: torch.Tensor, x: torch.Tensor, u0: torch.Tensor):
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

        self.proj_a = nn.Linear(input_dim, self.dimq - 1, bias=False)
        self.proj_z = nn.Linear(input_dim, self.dimq, bias=False)
        self.unproj = nn.Linear(self.dimq, input_dim, bias=False)

        self.pinn = PINN(
            matrix_dim=self.dimq,
            matrix_type="compressed",
            hidden_dims=list(pinn_hidden),
            t_domain=pinn_t_domain,
            activation=pinn_activation,
            dtype=torch.float32,
        )

        ckpt = torch.load(pinn_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.pinn.load_state_dict(state_dict)
        self.pinn.eval()
        for p in self.pinn.parameters():
            p.requires_grad = False

        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape
        if c != self.input_dim:
            raise ValueError(f"BuNNLayerStrict expected dim {self.input_dim}, got {c}")

        a = self.proj_a(x)
        z = self.proj_z(x)
        t_batch = torch.full((n, 1), self.t, dtype=x.dtype, device=x.device)
        y = self.pinn(t=t_batch, x=a, u0=z)["ut_theta"]
        out = self.unproj(y)
        return out


class LightGCNBuNNRecommender(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 64,
        n_layers: int = 3,
        use_bunn: bool = False,
        bunn_beta: float = 0.005,
        pinn_ckpt_path: str = "model_best.pth",
    ):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.use_bunn = bool(use_bunn)
        self.bunn_beta = float(bunn_beta)

        self.user_embeddings = nn.Embedding(self.n_users, self.dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        self.bunn = BuNNLayerStrict(
            input_dim=self.dim,
            t=1.0,
            matrix_dim=min(64, self.dim),
            pinn_ckpt_path=pinn_ckpt_path,
        )

    def full_embeddings(self) -> torch.Tensor:
        return torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

    def encode_lightgcn(self, edge_index: torch.Tensor) -> torch.Tensor:
        z0 = self.full_embeddings()
        edge_index_ud = make_undirected(edge_index)

        zs = [z0]
        z = z0
        for _ in range(self.n_layers):
            z = lightgcn_propagate(z, edge_index_ud)
            zs.append(z)

        z_lgcn = torch.stack(zs, dim=0).mean(dim=0)
        return z_lgcn

    def encode(self, edge_index: torch.Tensor):
        z_lgcn = self.encode_lightgcn(edge_index)

        if not self.use_bunn:
            loss_smap = torch.tensor(0.0, dtype=z_lgcn.dtype, device=z_lgcn.device)
            return z_lgcn, loss_smap

        z_bunn = self.bunn(z_lgcn.float()).to(z_lgcn.dtype)
        z_bunn = F.normalize(z_bunn, dim=1)

        delta = self.bunn_beta * z_bunn
        z_final = z_lgcn + delta
        loss_smap = torch.mean(delta ** 2)

        return z_final, loss_smap

    def split_embeddings(self, z: torch.Tensor):
        user_z = z[:self.n_users]
        item_z = z[self.n_users:self.n_users + self.n_items]
        return user_z, item_z

    def score_pairs(self, z: torch.Tensor, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_z, item_z = self.split_embeddings(z)
        zu = F.normalize(user_z[users], dim=1)
        zi = F.normalize(item_z[items], dim=1)
        return torch.sum(zu * zi, dim=1)

    def bpr_loss(
        self,
        edge_index: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        alpha_smap: float = 0.0,
    ):
        z, loss_smap = self.encode(edge_index)

        pos_scores = self.score_pairs(z, users, pos_items)
        neg_scores = self.score_pairs(z, users, neg_items)

        diff = pos_scores - neg_scores
        loss_bpr = -torch.mean(F.logsigmoid(diff))
        loss = loss_bpr + alpha_smap * loss_smap

        with torch.no_grad():
            pair_acc = (diff > 0).float().mean().item()
            pos_mean = pos_scores.mean().item()
            neg_mean = neg_scores.mean().item()

        return loss, {
            "loss_bpr": float(loss_bpr.item()),
            "loss_smap": float(loss_smap.item()),
            "pair_acc": float(pair_acc),
            "pos_mean": float(pos_mean),
            "neg_mean": float(neg_mean),
        }

    @torch.no_grad()
    def recommend_topk(self, edge_index: torch.Tensor, user_ids: torch.Tensor, k: int = 20, seen_items=None):
        z, _ = self.encode(edge_index)
        user_z, item_z = self.split_embeddings(z)

        user_z = F.normalize(user_z, dim=1)
        item_z = F.normalize(item_z, dim=1)

        scores = user_z[user_ids] @ item_z.t()

        result = {}
        for row, user_id in enumerate(user_ids.tolist()):
            row_scores = scores[row].clone()
            if seen_items is not None and user_id in seen_items:
                seen = list(seen_items[user_id])
                if len(seen) > 0:
                    row_scores[torch.tensor(seen, device=row_scores.device)] = -1e18
            topk = torch.topk(row_scores, k=k).indices.cpu().tolist()
            result[user_id] = topk
        return result


@torch.no_grad()
def evaluate_model(
    model: LightGCNBuNNRecommender,
    edge_index: torch.Tensor,
    device: torch.device,
    train_seen,
    test_pos,
    topk: int = 20,
):
    model.eval()
    users = sorted([u for u in test_pos.keys() if len(test_pos[u]) > 0])
    if len(users) == 0:
        return {f"Recall@{topk}": 0.0, f"NDCG@{topk}": 0.0}

    user_ids = torch.as_tensor(users, dtype=torch.long, device=device)
    recs = model.recommend_topk(edge_index=edge_index, user_ids=user_ids, k=topk, seen_items=train_seen)

    recalls = []
    ndcgs = []
    for u in users:
        gt = set(test_pos[u])
        recall, ndcg = recall_ndcg_at_k(recs[u], gt, topk)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        f"Recall@{topk}": float(np.mean(recalls)),
        f"NDCG@{topk}": float(np.mean(ndcgs)),
    }


def train_one_epoch(
    model: LightGCNBuNNRecommender,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    user_pos_items,
    n_items: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    alpha_smap: float,
):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_smap = 0.0

    for _ in range(n_batches):
        users, pos_items, neg_items = sample_bpr_batch(user_pos_items, n_items, batch_size, device)

        optimizer.zero_grad()
        loss, stats = model.bpr_loss(
            edge_index=edge_index,
            users=users,
            pos_items=pos_items,
            neg_items=neg_items,
            alpha_smap=alpha_smap,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += stats["pair_acc"]
        total_pos += stats["pos_mean"]
        total_neg += stats["neg_mean"]
        total_smap += stats["loss_smap"]

    denom = float(n_batches)
    return {
        "loss": total_loss / denom,
        "pair_acc": total_acc / denom,
        "pos_mean": total_pos / denom,
        "neg_mean": total_neg / denom,
        "loss_smap": total_smap / denom,
    }


def run_experiment(
    data_dir: str,
    feature_dim: int = 64,
    n_layers: int = 3,
    use_bunn: bool = False,
    bunn_beta: float = 0.005,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    epochs: int = 20,
    batch_size: int = 2048,
    n_batches: int = 100,
    alpha_smap: float = 1e-3,
    topk: int = 20,
    seed: int = 42,
    pinn_ckpt_path: str = "model_best.pth",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_gowalla_or_yelp2018(data_dir)
    n_users = data["n_users"]
    n_items = data["n_items"]
    edge_index = data["edge_index"].to(device)

    model = LightGCNBuNNRecommender(
        n_users=n_users,
        n_items=n_items,
        dim=feature_dim,
        n_layers=n_layers,
        use_bunn=use_bunn,
        bunn_beta=bunn_beta,
        pinn_ckpt_path=pinn_ckpt_path,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_recall = -1.0
    best_metrics = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            edge_index=edge_index,
            user_pos_items=data["train_user_pos"],
            n_items=n_items,
            batch_size=batch_size,
            n_batches=n_batches,
            device=device,
            alpha_smap=alpha_smap,
        )

        metrics = evaluate_model(
            model=model,
            edge_index=edge_index,
            device=device,
            train_seen=data["train_seen"],
            test_pos=data["test_user_pos"],
            topk=topk,
        )

        recall_key = f"Recall@{topk}"
        if metrics[recall_key] > best_recall:
            best_recall = metrics[recall_key]
            best_metrics = dict(metrics)
            best_epoch = epoch

        print(
            f"epoch={epoch} "
            f"loss={train_stats['loss']:.6f} "
            f"smap={train_stats['loss_smap']:.6f} "
            f"pair_acc={train_stats['pair_acc']:.4f} "
            f"pos_mean={train_stats['pos_mean']:.4f} "
            f"neg_mean={train_stats['neg_mean']:.4f} "
            f"Recall@{topk}={metrics[f'Recall@{topk}']:.4f} "
            f"NDCG@{topk}={metrics[f'NDCG@{topk}']:.4f}"
        )

    print("best_epoch =", best_epoch)
    print("best_metrics =", best_metrics)

    u = list(data["test_user_pos"].keys())[0]
    print("example_user =", u)
    print("train_items_sample =", data["train_user_pos"].get(u, [])[:10])
    print("test_items_sample =", data["test_user_pos"].get(u, [])[:10])

    user_ids = torch.tensor([u], dtype=torch.long, device=device)
    recs = model.recommend_topk(edge_index=edge_index, user_ids=user_ids, k=topk, seen_items=data["train_seen"])
    print("top20 =", recs[u])

    return model, best_metrics


if __name__ == "__main__":
    dataset_name = "gowalla"
    data_dir = download_builtin_dataset(dataset_name, root_dir="./data")

    run_experiment(
        data_dir=data_dir,
        feature_dim=64,
        n_layers=1,
        use_bunn=True,
        bunn_beta=0.005,
        lr=1e-4,
        weight_decay=5e-4,
        epochs=20,
        batch_size=2048,
        n_batches=100,
        alpha_smap=1e-3,
        topk=20,
        seed=42,
        pinn_ckpt_path="model_best.pth",
    )