import argparse
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.io import mmread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from split import edge_split, build_train_graph, assert_no_overlap


# ============================================================
# Graph loading (same datasets, same logic)
# ============================================================

def load_graph_from_mtx(path: str) -> nx.Graph:
    """
    Load an undirected collaboration graph from a .mtx file.
    """
    A = mmread(path).tocsr()
    G = nx.from_scipy_sparse_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    return G


# ============================================================
# Convert NetworkX graph to PyG Data object
# ============================================================

def nx_to_pyg(G: nx.Graph):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.

    - Nodes are reindexed from 0..N-1 internally
    - We keep a mapping to recover original node IDs later
    """
    nodes = list(G.nodes())
    node_id_map = {nid: i for i, nid in enumerate(nodes)}

    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_id_map[u], node_id_map[v]])
        edge_index.append([node_id_map[v], node_id_map[u]])  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Simple node features:
    # here we use node degree as a 1D feature
    x = torch.tensor(
        [[G.degree(n)] for n in nodes],
        dtype=torch.float
    )

    data = Data(x=x, edge_index=edge_index)
    return data, node_id_map


# ============================================================
# GraphSAGE model
# ============================================================

class GraphSAGE(nn.Module):
    """
    2-layer GraphSAGE model.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================
# Training loop (unsupervised via link reconstruction)
# ============================================================

def main():
    """
    GraphSAGE pipeline (train-only):
    1) Load full graph
    2) Split edges
    3) Build G_train using training edges only
    4) Convert G_train to PyG format
    5) Train GraphSAGE
    6) Save node embeddings
    """

    # -----------------------------
    # CLI arguments
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ca-CondMat", "ca-GrQc", "ca-HepPh"])
    ap.add_argument("--outdir", default="artifacts/embeddings")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # -----------------------------
    # 1) Load full graph
    # -----------------------------
    mtx = f"data/{args.dataset}/{args.dataset}.mtx"
    G_full = load_graph_from_mtx(mtx)

    # -----------------------------
    # 2) Split edges
    # -----------------------------
    train_e, val_e, test_e = edge_split(G_full, seed=args.seed)
    assert_no_overlap(train_e, val_e, test_e)

    # -----------------------------
    # 3) Build TRAIN-ONLY graph
    # -----------------------------
    G_train = build_train_graph(G_full, train_e)

    # -----------------------------
    # 4) Convert to PyG format
    # -----------------------------
    data, node_id_map = nx_to_pyg(G_train)

    # -----------------------------
    # 5) Initialize GraphSAGE
    # -----------------------------
    model = GraphSAGE(
        in_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------
    # 6) Train (simple unsupervised objective)
    # -----------------------------
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        # Simple regularization loss (L2 norm)
        loss = (z ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

    # -----------------------------
    # 7) Save embeddings
    # -----------------------------
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index).cpu().numpy()

    # Map back to original node IDs
    inv_map = {v: k for k, v in node_id_map.items()}
    emb = {inv_map[i]: z[i] for i in range(z.shape[0])}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"{args.dataset}_graphsage_d{args.out_dim}.npy"
    np.save(out_path, emb, allow_pickle=True)

    print("Saved:", out_path)
    print("Num nodes embedded:", len(emb))


if __name__ == "__main__":
    main()
# ============================================================