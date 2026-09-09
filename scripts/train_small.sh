#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m machinezero.training.train --out checkpoints/model.pt
