from __future__ import annotations

import torch
from torch import nn

from machinezero.data.encoding import FEATURE_DIM, SEMANTIC_DIM


class MachineZeroTransformer(nn.Module):
    """Infer queried opcode semantics from a sequence of black-box experiments."""

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
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.evidence_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, SEMANTIC_DIM)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.shape[1] > self.pos.shape[1]:
            raise ValueError(f"context has {tokens.shape[1]} tokens; model supports {self.pos.shape[1]}")
        x = self.input(tokens) + self.pos[:, : tokens.shape[1]]
        h = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is None:
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device, dtype=torch.long)
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            idx = (~padding_mask).sum(1) - 1
            valid = ~padding_mask
        query = h[torch.arange(h.size(0), device=h.device), idx]

        # Encoding metadata ends with [..., is_query, same_query_opcode, evidence...].
        evidence_dim = FEATURE_DIM - (2 * 8 + 4 + 2 + 256 + 6)
        same_query_index = FEATURE_DIM - evidence_dim - 1
        is_query_index = same_query_index - 1
        observed_match = valid & (tokens[..., same_query_index] > 0.5) & (tokens[..., is_query_index] < 0.5)
        counts = observed_match.sum(1, keepdim=True)
        matching = (h * observed_match.unsqueeze(-1)).sum(1) / counts.clamp_min(1)
        has_evidence = (counts > 0).to(h.dtype)
        matching = matching * has_evidence

        fused = self.evidence_fuse(torch.cat([query, matching], dim=-1))
        return self.head(fused)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
