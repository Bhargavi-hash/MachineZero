#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python -m machinezero.evaluation.baseline_eval --out results/baselines.json
python -m machinezero.evaluation.system_id_eval --out results/system_id_eval.json
if [[ -f checkpoints/model.pt ]]; then
  python -m machinezero.evaluation.evaluate --checkpoint checkpoints/model.pt --out results/eval.json
fi
