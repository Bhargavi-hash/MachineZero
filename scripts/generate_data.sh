#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m machinezero.data.generate --config configs/data/small.yaml
