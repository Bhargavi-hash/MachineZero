from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable pretrained MachineZero model."""

    name: str
    repo_id: str
    checkpoint_name: str = "model.pt"


OFFICIAL_MODELS: dict[str, ModelSpec] = {
    "small": ModelSpec(
        name="small",
        repo_id=os.environ.get("MACHINEZERO_SMALL_REPO", "Bhargavi-Kurukunda/machinezero-small"),
    ),
}


def model_cache_root() -> Path:
    if override := os.environ.get("MACHINEZERO_MODEL_HOME"):
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "machinezero" / "models"


def resolve_model_spec(name_or_repo: str) -> ModelSpec:
    if name_or_repo in OFFICIAL_MODELS:
        return OFFICIAL_MODELS[name_or_repo]
    if "/" in name_or_repo:
        return ModelSpec(name=name_or_repo.rsplit("/", 1)[-1], repo_id=name_or_repo)
    choices = ", ".join(sorted(OFFICIAL_MODELS))
    raise ValueError(
        f"unknown model {name_or_repo!r}; choose one of [{choices}] or pass a Hugging Face repo ID like owner/repo"
    )


def model_directory(name_or_repo: str) -> Path:
    spec = resolve_model_spec(name_or_repo)
    return model_cache_root() / spec.name


def model_checkpoint(name_or_repo: str) -> Path:
    spec = resolve_model_spec(name_or_repo)
    return model_directory(name_or_repo) / spec.checkpoint_name


def pull_model(name_or_repo: str, force: bool = False) -> tuple[ModelSpec, Path]:
    from huggingface_hub import snapshot_download

    spec = resolve_model_spec(name_or_repo)
    destination = model_directory(name_or_repo)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.repo_id, local_dir=destination, force_download=force)
    checkpoint = destination / spec.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"downloaded {spec.repo_id}, but {spec.checkpoint_name!r} was not found in the model repository"
        )
    return spec, checkpoint


def resolve_checkpoint(*, checkpoint: str = "", model: str = "") -> Path:
    if checkpoint and model:
        raise ValueError("use either --checkpoint or --model, not both")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if model:
        path = model_checkpoint(model)
        if not path.is_file():
            raise FileNotFoundError(f"model {model!r} is not downloaded; run `machinezero pull {model}` first")
        return path
    raise ValueError("provide either --checkpoint PATH or --model NAME")
