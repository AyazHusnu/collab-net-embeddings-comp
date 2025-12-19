import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.io import mmread
from node2vec import Node2Vec

from split import edge_split, build_train_graph, assert_no_overlap


# ============================================================
# Graph loading (same as DeepWalk)
# ============================================================

def load_graph_from_mtx(path: str) -> nx.Graph:
    """
    Load an undirected simple graph from a Matrix Market (.mtx) adjacency matrix.
    """
    A = mmread(path).tocsr()
    G = nx.from_scipy_sparse_array(A)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Ensure simple undirected graph
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)

    return G


# ============================================================
# Node2Vec training
# ============================================================

def main():
    """
    Node2Vec pipeline (train-only):
    1) Load full graph (for splitting only)
    2) Split edges into train/val/test
    3) Build G_train using ONLY training edges
    4) Fit Node2Vec on G_train (biased random walks)
    5) Save embeddings (node_id -> vector)

    Key rule to avoid leakage:
    - Node2Vec is trained ONLY on G_train (no val/test edges).
    """

    # -----------------------------
    # CLI arguments
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ca-CondMat", "ca-GrQc", "ca-HepPh"])
    ap.add_argument("--outdir", default="artifacts/embeddings")
    ap.add_argument("--seed", type=int, default=42)

    # Embedding hyperparams
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walk_length", type=int, default=80)
    ap.add_argument("--num_walks", type=int, default=10)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)

    # Node2Vec-specific hyperparams:
    # p  : "return parameter"  (high p -> less likely to immediately go back)
    # q  : "in-out parameter"  (low q -> explore outward; high q -> stay local)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)

    args = ap.parse_args()

    # -----------------------------
    # 1) Load full graph
    # -----------------------------
    mtx_path = f"data/{args.dataset}/{args.dataset}.mtx"
    G_full = load_graph_from_mtx(mtx_path)

    # -----------------------------
    # 2) Split edges (same strategy as Person 1)
    # -----------------------------
    train_e, val_e, test_e = edge_split(G_full, seed=args.seed)
    assert_no_overlap(train_e, val_e, test_e)

    # -----------------------------
    # 3) Build training-only graph
    # -----------------------------
    # This is the critical step: embeddings must not use future edges.
    G_train = build_train_graph(G_full, train_e)

    # -----------------------------
    # 4) Fit Node2Vec on TRAIN graph only
    # -----------------------------
    # Node2Vec generates biased random walks on G_train, then trains a skip-gram model.
    n2v = Node2Vec(
        G_train,
        dimensions=args.dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=args.workers,
        seed=args.seed,
    )

    # Fit the internal Word2Vec model
    model = n2v.fit(
        window=args.window,
        min_count=0,
        epochs=args.epochs,
        batch_words=2048,
    )

    # -----------------------------
    # 5) Save embeddings
    # -----------------------------
    # Same format as DeepWalk: dict[int -> np.array]
    emb = {int(k): model.wv[k] for k in model.wv.index_to_key}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"{args.dataset}_node2vec_d{args.dim}_p{args.p}_q{args.q}.npy"
    np.save(out_path, emb, allow_pickle=True)

    print("Saved:", out_path)
    print("Num nodes embedded:", len(emb))


if __name__ == "__main__":
    main()
# ============================================================