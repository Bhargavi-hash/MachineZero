#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python demo/demo.py --seed "${1:-2026}" --budget "${2:-20}"
