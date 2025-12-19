#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="LFN"

# Go to repo root (where this script is)
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

echo "[1/4] Creating conda env from environment.yml (name: $ENV_NAME)"
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda env create -f environment.yml

echo "[2/4] Activating env"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[3/4] Quick import checks"
python - <<'PY'
import networkx, numpy, pandas, sklearn
print("core libs OK")
try:
    import gensim, tqdm
    from node2vec import Node2Vec
    import torch
    from torch_geometric.nn import SAGEConv
    print("embeddings libs OK")
except Exception as e:
    raise SystemExit(f"Missing embedding libs: {e}")
PY

echo "[4/4] Done."
echo "To use it now: conda activate $ENV_NAME"
SH
chmod +x setup_env.sh