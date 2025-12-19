# Collaboration Networks & Embedding Comparison

A comprehensive study comparing node embedding methods for link prediction on collaboration networks.

# Embedding Models (Enzo)

This repository contains **train-only node embedding pipelines** used for link prediction.
All embeddings are trained **only on the training graph** (`G_train`) produced by the split logic,
so **no validation/test edges are used during training**.

## What was implemented
- **DeepWalk** (random walks + skip-gram)
- **Node2Vec** (biased random walks with parameters `p`, `q`)
- **GraphSAGE** (neural message passing)

All scripts:
- reuse the same **topology-based split** as Ayaz
- build **`G_train` with all nodes and training edges only**
- save embeddings in a **common format** for Sinem

## Environment setup
Create and activate the environment (reproducible):
```bash
./setup_env.sh
conda activate LFN
```

## How to generate embeddings

Run commands **from the repository root**.

### DeepWalk
```bash
PYTHONPATH=src python src/train_deepwalk.py   --dataset ca-GrQc   --dim 64   --num_walks 10   --walk_length 80   --window 10   --epochs 5
```

### Node2Vec
```bash
PYTHONPATH=src python src/train_node2vec.py   --dataset ca-GrQc   --dim 64   --num_walks 10   --walk_length 80   --window 10   --epochs 5   --p 1.0   --q 1.0
```

### GraphSAGE
```bash
PYTHONPATH=src python src/train_graphsage.py   --dataset ca-GrQc   --out_dim 64   --epochs 50
```

Supported datasets:
- `ca-CondMat`
- `ca-GrQc`
- `ca-HepPh`

## Outputs
Embeddings are saved to:
```
artifacts/embeddings/
```
Examples:
- `ca-GrQc_deepwalk_d64.npy`
- `ca-GrQc_node2vec_d64_p1.0_q1.0.npy`
- `ca-GrQc_graphsage_d64.npy`

### File format (for Sinem)
Each `.npy` file contains a **Python dictionary**:
```python
{ node_id (int) : embedding (np.ndarray, shape = [dim]) }
```
This format is identical across DeepWalk, Node2Vec, and GraphSAGE.

> Note: Generated artifacts are intentionally **not versioned** (see `.gitignore`).
They are fully reproducible from the code and `environment.yml`.
