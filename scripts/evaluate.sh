#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m machinezero.evaluation.system_id_eval
[ -f checkpoints/model.pt ] && PYTHONPATH=src python -m machinezero.evaluation.evaluate --checkpoint checkpoints/model.pt || true
