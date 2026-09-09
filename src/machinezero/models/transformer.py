from __future__ import annotations

import torch
from torch import nn

from machinezero.data.encoding import FEATURE_DIM, SEMANTIC_DIM


class MachineZeroTransformer(nn.Module):
    """Infer the queried opcode's latent semantics from observed transitions."""

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        max_tokens: int = 31,
    ):
        super().__init__()
        self.d_model = d_model
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, SEMANTIC_DIM))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.shape[1] > self.pos.shape[1]:
            raise ValueError(f'context has {tokens.shape[1]} tokens; model supports {self.pos.shape[1]}')
        x = self.input(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            idx = (~padding_mask).sum(1) - 1
            valid = ~padding_mask
        q = h[torch.arange(h.size(0), device=h.device), idx]

        # Last two encoded features are is_query and same_query_opcode. Pool only
        # *observed* transitions matching the queried opcode. This makes exact
        # opcode identity a retrieval key without exposing hidden semantics.
        observed_match = valid & (tokens[..., -1] > 0.5) & (tokens[..., -2] < 0.5)
        counts = observed_match.sum(1, keepdim=True)
        evidence = (h * observed_match.unsqueeze(-1)).sum(1) / counts.clamp_min(1)
        has_evidence = (counts > 0).to(h.dtype)
        q = q + evidence * has_evidence
        return self.head(q)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
