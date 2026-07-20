#!/usr/bin/env python3
"""
Graph-RoPE: Graph-Aware Rotary Position Embeddings.

Replaces standard sequential RoPE in ModernBERT with structural
positional encoding derived from CPG features (Poisson potential +
multi-scale diffusion coordinates).

Design E1 (Non-Strict): All heads share the same cos/sin values.
Each of the 6 graph feature dimensions is assigned a bank of
rotation-plane frequencies.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ── Default corpus statistics (from analysis of v0.4 shard sample) ──────
# These are p5/p95 percentiles per feature dimension.
# Used for normalizing raw graph features to [0, max_pos].
DEFAULT_FEAT_P5 = [-1.215, -19.441, -19.376, -18.918, -18.918, -18.918]
DEFAULT_FEAT_P95 = [1.679, -3.314, -3.178, -2.909, -2.890, -2.890]


class GraphRoPEConfig:
    """Configuration for Graph-RoPE."""

    def __init__(
        self,
        d_head: int = 64,
        n_features: int = 6,
        max_position: float = 2048.0,
        base: float = 10000.0,
        # Feature-to-plane assignment: (feature_dim, n_planes)
        # E1 Non-Strict: same for all heads
        plane_assignment: Optional[list[tuple[int, int]]] = None,
        feat_p5: Optional[list[float]] = None,
        feat_p95: Optional[list[float]] = None,
    ):
        self.d_head = d_head
        self.n_features = n_features
        self.max_position = max_position
        self.base = base
        self.feat_p5 = feat_p5 or DEFAULT_FEAT_P5
        self.feat_p95 = feat_p95 or DEFAULT_FEAT_P95

        # Default plane assignment: 6 planes for Poisson, 5 each for
        # diffusion scales, 6 for last scale = 32 total = d_head / 2
        if plane_assignment is None:
            self.plane_assignment = [
                (0, 6),   # Poisson:  planes 0..5
                (1, 5),   # d_scale1: planes 6..10
                (2, 5),   # d_scale2: planes 11..15
                (3, 5),   # d_scale3: planes 16..20
                (4, 5),   # d_scale4: planes 21..25
                (5, 6),   # d_scale5: planes 26..31
            ]
        else:
            self.plane_assignment = plane_assignment

        # Verify total planes = d_head / 2
        total_planes = sum(n for _, n in self.plane_assignment)
        assert total_planes == d_head // 2, (
            f"Plane assignment sums to {total_planes}, "
            f"expected {d_head // 2}"
        )


class GraphRoPEEmbedding(nn.Module):
    """
    Computes cos/sin rotary embeddings from 6D graph features.

    Input:  graph_features [B, T, 6]
    Output: (cos, sin) each [B, T, d_head/2]

    The cos/sin are identical across all heads in E1 (Non-Strict mode).
    """

    def __init__(self, config: GraphRoPEConfig):
        super().__init__()
        self.config = config
        self.d_half = config.d_head // 2

        # ── Register normalization stats as buffers (not parameters) ───
        self.register_buffer(
            "feat_p5",
            torch.tensor(config.feat_p5, dtype=torch.float32),
        )
        self.register_buffer(
            "feat_p95",
            torch.tensor(config.feat_p95, dtype=torch.float32),
        )
        feat_range = self.feat_p95 - self.feat_p5
        # Avoid division by zero for constant features
        feat_range = feat_range.clamp(min=1e-6)
        self.register_buffer("feat_range", feat_range)

        # ── Build inverse-frequency banks ─────────────────────────────
        # Each feature dimension gets its own bank of frequencies.
        # Within a bank, frequencies follow geometric progression:
        #   θ_j = base^{-2j / n_planes}
        inv_freq_parts = []
        for feat_dim, n_planes in config.plane_assignment:
            exponents = torch.arange(0, n_planes, dtype=torch.float32)
            # Standard RoPE frequency formula adapted per-bank
            bank = 1.0 / (
                config.base ** (2.0 * exponents / (2.0 * n_planes))
            )
            inv_freq_parts.append(bank)

        # Concatenate all banks → [d_head/2]
        inv_freq = torch.cat(inv_freq_parts, dim=0)
        assert inv_freq.shape[0] == self.d_half
        self.register_buffer("inv_freq", inv_freq)

        # ── Build feature-index mapping ───────────────────────────────
        # feat_idx[plane] = which feature dimension drives this plane
        feat_idx = []
        for feat_dim, n_planes in config.plane_assignment:
            feat_idx.extend([feat_dim] * n_planes)
        self.register_buffer(
            "feat_idx",
            torch.tensor(feat_idx, dtype=torch.long),
        )

    def normalize_features(
        self, graph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize raw graph features → [0, max_position].

        Clips to [p5, p95] range first for outlier robustness.

        Args:
            graph_features: [B, T, 6] raw CPG features
        Returns:
            [B, T, 6] normalized to [0, max_pos]
        """
        # Clip to robust range
        clipped = graph_features.clamp(
            min=self.feat_p5, max=self.feat_p95
        )
        # Affine normalization: [p5, p95] → [0, max_pos]
        normalized = (
            (clipped - self.feat_p5) / self.feat_range
            * self.config.max_position
        )
        return normalized

    def forward(
        self, graph_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos/sin rotary embeddings from graph features.

        Args:
            graph_features: [B, T, 6] raw CPG features per token

        Returns:
            (cos, sin): each [B, T, d_head/2]
        """
        # 1. Normalize features → [0, max_pos]
        normed = self.normalize_features(graph_features)  # [B, T, 6]

        # 2. Select the feature value for each rotation plane
        #    feat_idx maps each of 32 planes to one of 6 features
        per_plane = normed[..., self.feat_idx]  # [B, T, d_half]

        # 3. Multiply by per-plane inverse frequency
        #    angles[b, t, plane] = inv_freq[plane] * φ_normalized[b, t, plane]
        angles = per_plane * self.inv_freq  # [B, T, d_half]

        # 4. Compute cos/sin for rotary application
        cos = angles.cos()
        sin = angles.sin()

        return cos, sin


# ── Rotary application functions ─────────────────────────────────────────
# These are identical to standard RoPE application — we only change
# how cos/sin are computed (graph-structural instead of sequential).

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [x1, x2, x3, x4] → [-x2, x1, -x4, x3]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_graph_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: [B, n_heads, T, d_head]
        k: [B, n_heads, T, d_head]
        cos: [B, T, d_head/2]  (from GraphRoPEEmbedding)
        sin: [B, T, d_head/2]

    Returns:
        (q_rot, k_rot) with same shapes as input
    """
    # Expand cos/sin for all heads: [B, T, d/2] → [B, 1, T, d/2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Duplicate cos/sin across the full d_head:
    # [B, 1, T, d/2] → [B, 1, T, d]
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot
