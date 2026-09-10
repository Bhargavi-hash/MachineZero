import torch

from machinezero.data.encoding import FEATURE_DIM, SEMANTIC_DIM
from machinezero.models.transformer import MachineZeroTransformer


def test_model_shape():
    model = MachineZeroTransformer(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_tokens=6,
    )
    y = model(
        torch.randn(2, 6, FEATURE_DIM),
        torch.zeros(2, 6, dtype=torch.bool),
    )
    assert y.shape == (2, SEMANTIC_DIM)
