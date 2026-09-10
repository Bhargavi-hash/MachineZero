from __future__ import annotations

import torch
from torch.nn import functional as F

from machinezero.data.encoding import OPERATIONS


def semantic_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Joint operation classification and semantic-attribute loss."""
    n_ops = len(OPERATIONS)
    operation = F.cross_entropy(logits[:, :n_ops], target[:, 0].long())
    attributes = F.binary_cross_entropy_with_logits(logits[:, n_ops:], target[:, 1:].float())
    return operation + 0.5 * attributes
