from pathlib import Path

import pytest

from machinezero.models.registry import model_cache_root, model_checkpoint, resolve_checkpoint, resolve_model_spec


def test_small_model_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINEZERO_MODEL_HOME", str(tmp_path))
    spec = resolve_model_spec("small")
    assert spec.name == "small"
    assert spec.repo_id
    assert model_cache_root() == tmp_path
    assert model_checkpoint("small") == tmp_path / "small" / "model.pt"


def test_arbitrary_hugging_face_repo_is_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINEZERO_MODEL_HOME", str(tmp_path))
    spec = resolve_model_spec("someone/custom-machinezero")
    assert spec.name == "custom-machinezero"
    assert spec.repo_id == "someone/custom-machinezero"


def test_resolve_checkpoint_prefers_explicit_file(tmp_path):
    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_checkpoint(checkpoint=str(checkpoint)) == checkpoint


def test_resolve_checkpoint_requires_download(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINEZERO_MODEL_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="machinezero pull small"):
        resolve_checkpoint(model="small")


def test_checkpoint_and_model_are_mutually_exclusive(tmp_path):
    checkpoint = Path(tmp_path) / "local.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="either --checkpoint or --model"):
        resolve_checkpoint(checkpoint=str(checkpoint), model="small")
