# -----------------------------
# Standard library imports
# -----------------------------

import argparse
from pathlib import Path

# -----------------------------
# Scientific / ML libraries
# -----------------------------

import networkx as nx          # graph manipulation
import numpy as np             # numerical operations
from scipy.io import mmread    # load .mtx sparse matrices
from gensim.models import Word2Vec  # Skip-gram model used by DeepWalk
from tqdm import tqdm          # progress bars

# -----------------------------
# Project-specific utilities
# -----------------------------

# We reuse the exact same splitting logic as in the notebook,
# but in a clean, reusable Python module.
from split import edge_split, build_train_graph, assert_no_overlap


# ============================================================
# Graph loading
# ============================================================

def load_graph_from_mtx(path: str) -> nx.Graph:
    """
    Load a collaboration graph from a Matrix Market (.mtx) file.

    The matrix represents an undirected adjacency matrix.
    Each non-zero entry corresponds to a collaboration edge.

    Returns a simple undirected NetworkX graph.
    """
    A = mmread(path).tocsr()               # sparse adjacency matrix
    G = nx.from_scipy_sparse_array(A)      # build graph from adjacency

    # Remove self-loops (a node collaborating with itself)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Safety: convert to simple graph if needed
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)

    return G


# ============================================================
# DeepWalk random walk generation
# ============================================================

def generate_walks(G: nx.Graph, num_walks: int, walk_length: int, seed: int):
    """
    Generate random walks on the TRAINING graph only (G_train).

    Each walk is treated as a "sentence":
    nodes are tokens, and the walk captures graph proximity.

    IMPORTANT:
    - Walks are generated ONLY on G_train
    - No validation or test edges are visible here
    """
    rng = np.random.default_rng(seed)
    nodes = np.array(list(G.nodes()), dtype=int)
    walks = []

    # Repeat the process num_walks times
    for _ in tqdm(range(num_walks), desc="DeepWalk walks"):
        rng.shuffle(nodes)  # randomize starting order

        for start in nodes:
            walk = [int(start)]
            cur = int(start)

            # Perform a simple random walk
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(cur))
                if not neighbors:
                    break  # isolated node

                cur = int(rng.choice(neighbors))
                walk.append(cur)

            # Word2Vec expects strings, not integers
            walks.append([str(x) for x in walk])

    return walks


# ============================================================
# Main training pipeline
# ============================================================

def main():
    """
    End-to-end DeepWalk pipeline:
    1. Load full graph
    2. Split edges (train / val / test)
    3. Build training graph (edges = train only)
    4. Generate random walks on training graph
    5. Train Skip-gram model
    6. Save node embeddings
    """

    # -----------------------------
    # Command-line arguments
    # -----------------------------

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ca-CondMat", "ca-GrQc", "ca-HepPh"])
    ap.add_argument("--outdir", default="artifacts/embeddings")
    ap.add_argument("--seed", type=int, default=42)

    # DeepWalk hyperparameters
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num_walks", type=int, default=10)
    ap.add_argument("--walk_length", type=int, default=80)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)

    args = ap.parse_args()

    # -----------------------------
    # 1) Load full graph
    # -----------------------------

    # Full graph is loaded ONLY to perform the split
    mtx_path = f"data/{args.dataset}/{args.dataset}.mtx"
    G_full = load_graph_from_mtx(mtx_path)

    # -----------------------------
    # 2) Split edges
    # -----------------------------

    # Same topology-based split as Person 1
    train_e, val_e, test_e = edge_split(G_full, seed=args.seed)

    # Safety check: no overlap between splits
    assert_no_overlap(train_e, val_e, test_e)

    # -----------------------------
    # 3) Build training graph
    # -----------------------------

    # G_train contains:
    # - ALL nodes
    # - ONLY training edges
    #
    # This is the key step to avoid information leakage
    G_train = build_train_graph(G_full, train_e)

    # -----------------------------
    # 4) Generate DeepWalk walks
    # -----------------------------

    walks = generate_walks(
        G_train,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.seed
    )

    # -----------------------------
    # 5) Train Skip-gram (Word2Vec)
    # -----------------------------

    model = Word2Vec(
        sentences=walks,
        vector_size=args.dim,
        window=args.window,
        min_count=0,
        sg=1,             # skip-gram
        workers=4,
        negative=5,
        seed=args.seed,
        epochs=args.epochs,
    )

    # -----------------------------
    # 6) Save embeddings
    # -----------------------------

    # Embeddings are stored as:
    # { node_id (int) : embedding vector (np.array) }
    emb = {int(k): model.wv[k] for k in model.wv.index_to_key}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_path = outdir / f"{args.dataset}_deepwalk_d{args.dim}.npy"
    np.save(out_path, emb, allow_pickle=True)

    print("Saved:", out_path)
    print("Num nodes embedded:", len(emb))


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    main()
