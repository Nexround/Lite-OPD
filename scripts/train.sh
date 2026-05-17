#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/example_qwen3_1b7_8b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$ROOT"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export PYTHONPATH="$ROOT/src"

exec python -m liteopd.train.launcher \
  --config "$CONFIG" \
  --nproc-per-node "$NPROC_PER_NODE"
