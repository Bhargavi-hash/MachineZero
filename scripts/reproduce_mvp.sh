#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
mkdir -p checkpoints results

echo '==> 1/6 tests'
pytest -q | tee results/pytest.txt

echo '==> 2/6 training learned model'
python -m machinezero.training.train \
  --data-config configs/data/small.yaml \
  --model-config configs/model/small.yaml \
  --train-config configs/train/default.yaml \
  --out checkpoints/model.pt \
  --metrics-out results/train.json | tee results/train.txt

echo '==> 3/6 learned-model held-out evaluation'
python -m machinezero.evaluation.evaluate \
  --checkpoint checkpoints/model.pt \
  --queries-per-arch 40 \
  --out results/eval.json | tee results/eval.txt

echo '==> 4/6 non-neural baselines and system identification'
python -m machinezero.evaluation.baseline_eval \
  --queries-per-arch 40 \
  --out results/baselines.json | tee results/baselines.txt
python -m machinezero.evaluation.system_id_eval \
  --out results/system_id_eval.json | tee results/system_id_eval.txt

echo '==> 5/6 simulator benchmark'
python benchmarks/benchmark_simulator.py \
  --n 20000 \
  --out results/benchmark.json | tee results/benchmark.txt

echo '==> 6/6 deterministic contest demo'
python demo/demo.py --seed 2026 --budget 20 | tee results/demo.txt

echo '==> complete'
