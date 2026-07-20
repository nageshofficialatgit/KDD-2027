#!/usr/bin/env python3
"""
WIRE: Wave-Induced Rotary Encodings for Graphs.
Ref: "Rotary Position Encodings for Graphs" (Reid et al., 2025).

Implemented as a standalone structural projection module for ablation.
"""

import torch
import torch.nn as nn


class WIREConfig:
    """Configuration for WIRE Structural Projection."""

    def __init__(
        self,
        d_head: int = 64,
        n_features: int = 6,
        base: float = 10000.0,
    ):
        self.d_head = d_head
        self.n_features = n_features
        self.base = base


class WIREEmbedding(nn.Module):
    """
    WIRE Structural Rotary Embeddings.
    θ = Projection(StructuralCoords)
    """

    def __init__(self, config: WIREConfig):
        super().__init__()
        self.config = config
        self.d_half = config.d_head // 2

        # ── Learnable Frequency Projection ──────────────────────────
        # Per WIRE paper Eq. (2): θ_n = ω_n^⊤ r_i
        # Each row ω_n ∈ ℝ^m is a learnable frequency vector.
        # bias=True lets the model learn its own centering (replaces
        # hardcoded feat_means: W(x-μ) = Wx + b where b = -Wμ).
        self.proj = nn.Linear(config.n_features, self.d_half, bias=True)
        self._initialize_projection()

    def _initialize_projection(self):
        """
        Initialize ω_n weights using the E1 strategy (Neural Network based).
        This allows the network to naturally learn optimal combinations of 
        arbitrary N features to produce the rotation frequencies, 
        without being forced into handcrafted manual assignments.
        """
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self, graph_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # WIRE Eq. (2): θ_n = ω_n^⊤ r_i + b_n
        angles = self.proj(graph_features)
        return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_wire_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(1).repeat(1, 1, 1, 2)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)
