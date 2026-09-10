#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m machinezero.training.train \
  --data-config configs/data/small.yaml \
  --model-config configs/model/small.yaml \
  --train-config configs/train/default.yaml \
  --out checkpoints/model.pt \
  --metrics-out results/train.json
