"""
Split utilities for link prediction.

This file contains the exact logic used in the exploratory notebook
(dataset_loading_preprocessing.ipynb), rewritten as reusable Python functions.

IMPORTANT:
- The logic is identical to the notebook.
- This file exists only to avoid code duplication and ensure reproducibility.
"""

import random
import networkx as nx
from sklearn.model_selection import train_test_split


# ---------- small helper ----------

def canon_edge(u, v):
    """
    Return a canonical (ordered) representation of an undirected edge.
    Ensures (u, v) and (v, u) are treated as the same edge.
    """
    return (u, v) if u <= v else (v, u)


# ---------- edge splitting ----------

def edge_split(G, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Topology-based split of edges for link prediction.

    This is exactly the same procedure used in the notebook:
    - Randomly split edges into train / validation / test.
    - No temporal information is used.
    """
    edges = list(G.edges())

    # First split: train vs (val + test)
    train_edges, temp_edges = train_test_split(
        edges,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )

    # Second split: validation vs test
    val_size = val_ratio / (1.0 - train_ratio)
    val_edges, test_edges = train_test_split(
        temp_edges,
        train_size=val_size,
        random_state=seed,
        shuffle=True
    )

    return train_edges, val_edges, test_edges


# ---------- negative sampling ----------

def sample_negative_edges(
    G,
    num_samples,
    forbidden_edges,
    seed=42,
    max_tries=10_000_000
):
    """
    Sample negative edges (u, v) such that:
    - u != v
    - (u, v) is not an existing edge
    - (u, v) is not in forbidden_edges

    This is the same negative sampling strategy used in the notebook.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())

    existing = set(canon_edge(u, v) for u, v in G.edges())
    forbidden = set(canon_edge(u, v) for u, v in forbidden_edges) | existing

    negatives = set()
    tries = 0

    while len(negatives) < num_samples and tries < max_tries:
        u = rng.choice(nodes)
        v = rng.choice(nodes)

        if u == v:
            tries += 1
            continue

        e = canon_edge(u, v)
        if e in forbidden:
            tries += 1
            continue

        negatives.add(e)

    if len(negatives) < num_samples:
        raise RuntimeError(
            f"Could only sample {len(negatives)} negatives out of {num_samples}. "
            f"Try lowering the number of samples."
        )

    return list(negatives)


def build_negative_splits(G, splits, neg_ratio=1.0, seed=42):
    """
    Build negative samples for train / validation / test.

    Important design choice (same as notebook):
    - All positive edges from ALL splits are forbidden
      to avoid any leakage or overlap.
    """
    train_pos = [canon_edge(*e) for e in splits["train"]]
    val_pos   = [canon_edge(*e) for e in splits["val"]]
    test_pos  = [canon_edge(*e) for e in splits["test"]]

    all_pos = set(train_pos) | set(val_pos) | set(test_pos)

    n_train = int(len(train_pos) * neg_ratio)
    n_val   = int(len(val_pos)   * neg_ratio)
    n_test  = int(len(test_pos)  * neg_ratio)

    train_neg = sample_negative_edges(
        G, n_train, forbidden_edges=all_pos, seed=seed + 1
    )
    val_neg = sample_negative_edges(
        G, n_val, forbidden_edges=all_pos | set(train_neg), seed=seed + 2
    )
    test_neg = sample_negative_edges(
        G, n_test,
        forbidden_edges=all_pos | set(train_neg) | set(val_neg),
        seed=seed + 3
    )

    return {
        "train_neg": train_neg,
        "val_neg": val_neg,
        "test_neg": test_neg,
    }


# ---------- training graph ----------

def build_train_graph(G_full, train_edges):
    """
    Build the training graph G_train.

    Key rule:
    - Nodes: all nodes from the original graph
    - Edges: ONLY training edges

    This guarantees that no validation/test edges
    are visible during embedding training.
    """
    G_train = nx.Graph()
    G_train.add_nodes_from(G_full.nodes())
    G_train.add_edges_from(train_edges)
    return G_train


# ---------- sanity checks ----------

def assert_no_overlap(train_e, val_e, test_e):
    """
    Safety check: ensure that edge splits do not overlap.
    """
    tr = set(canon_edge(*e) for e in train_e)
    va = set(canon_edge(*e) for e in val_e)
    te = set(canon_edge(*e) for e in test_e)

    assert len(tr & va) == 0, "Train / Val overlap"
    assert len(tr & te) == 0, "Train / Test overlap"
    assert len(va & te) == 0, "Val / Test overlap"
