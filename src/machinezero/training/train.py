from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from machinezero.data.dataset import ProceduralTransitionDataset
from machinezero.data.splits import make_splits, validate_disjoint
from machinezero.models.checkpoint import save_checkpoint
from machinezero.models.transformer import MachineZeroTransformer

from .losses import semantic_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/small.yaml")
    parser.add_argument("--model-config", default="configs/model/small.yaml")
    parser.add_argument("--train-config", default="configs/train/default.yaml")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--metrics-out", default="results/train.json")
    args = parser.parse_args()

    data_config = yaml.safe_load(Path(args.data_config).read_text())
    model_config = yaml.safe_load(Path(args.model_config).read_text())
    train_config = yaml.safe_load(Path(args.train_config).read_text())
    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and not train_config.get("cpu_only", False) else "cpu")
    splits = make_splits(
        data_config["base_seed"],
        data_config["train_architectures"],
        data_config["validation_architectures"],
        data_config["test_architectures"],
    )
    validate_disjoint(splits)
    dataset = ProceduralTransitionDataset(
        splits.train,
        data_config["samples_per_architecture"],
        data_config["max_context"],
        train_config["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    model = MachineZeroTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    print(f"device={device} parameters={model.parameter_count():,} samples={len(dataset)}")

    history = []
    model.train()
    for epoch in range(train_config["epochs"]):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["tokens"].to(device), batch["padding_mask"].to(device))
            loss = semantic_loss(logits, batch["target"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        epoch_loss = total / len(loader)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_config": model_config,
        "data_config": data_config,
        "train_config": train_config,
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "train_architecture_seeds": list(splits.train),
        "validation_architecture_seeds": list(splits.validation),
        "test_architecture_seeds": list(splits.test),
        "training_history": history,
    }
    save_checkpoint(args.out, model, metadata)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metadata, indent=2))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
