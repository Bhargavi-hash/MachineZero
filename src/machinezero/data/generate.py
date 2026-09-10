from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .splits import make_splits, validate_disjoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/small.yaml")
    p.add_argument("--out", default="results/data_metadata.json")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    s = make_splits(
        cfg["base_seed"], cfg["train_architectures"], cfg["validation_architectures"], cfg["test_architectures"]
    )
    validate_disjoint(s)
    meta = {
        "config": cfg,
        "splits": {"train_seeds": list(s.train), "validation_seeds": list(s.validation), "test_seeds": list(s.test)},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
