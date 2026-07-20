#!/usr/bin/env python3
"""
Graph-RoPE ModernBERT Model Wrapper.

Wraps pretrained ModernBERT-base with:
  - Graph-RoPE replacing standard sequential RoPE
  - Multi-task prediction heads (MLM, TFR, ERR/EdgePred)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ModernBertModel,
    ModernBertConfig,
    AutoTokenizer,
)

from .graph_rope import (
    GraphRoPEConfig,
    GraphRoPEEmbedding,
)

print(">>> BTP_TELEMETRY: models/graph_rope_model.py loaded with Sigmoid TFR head fix")


class GraphRoPEWrapper(nn.Module):
    """
    Wrapper that mimics ModernBertRotaryEmbedding's interface but
    returns pre-computed Graph-RoPE cos/sin.

    ModernBERT's attention calls:
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)

    Where cos/sin have shape [B, T, d_head] (full d_head, not half).
    Our Graph-RoPE produces [B, T, d_head/2], so we cat to match.
    """

    def __init__(self):
        super().__init__()
        self._graph_cos = None  # Set before forward pass
        self._graph_sin = None

    def set_cos_sin(self, cos: torch.Tensor, sin: torch.Tensor):
        """Store pre-computed cos/sin from GraphRoPEEmbedding."""
        self._graph_cos = cos
        self._graph_sin = sin

    def forward(self, x, position_ids=None, **kwargs):
        """
        Mimic ModernBertRotaryEmbedding.forward interface.

        Returns (cos, sin) each [B, T, d_head].
        The standard RoPE returns [B, T, d_head] by concatenating
        (freqs, freqs) — we do the same with our Graph-RoPE output.
        """
        if self._graph_cos is None:
            raise RuntimeError(
                "GraphRoPEWrapper: cos/sin not set. "
                "Call set_cos_sin() before forward pass."
            )
        # Our cos/sin are [B, T, d_head/2]
        # Standard ModernBERT expects [B, T, d_head] (doubled)
        cos = torch.cat([self._graph_cos, self._graph_cos], dim=-1)
        sin = torch.cat([self._graph_sin, self._graph_sin], dim=-1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GraphRoPEModernBERT(nn.Module):
    """
    ModernBERT with Graph-RoPE for multi-task pretraining.

    Architecture:
        - Pretrained ModernBERT encoder (RoPE replaced with Graph-RoPE)
        - MLM head (predict masked tokens)
        - Node type head (predict structural role)
        - Edge prediction head (predict edge existence between token pairs)
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        rope_config: GraphRoPEConfig = None,
        num_node_types: int = 5,
        mlm_weight: float = 1.0,
        node_weight: float = 0.5,
        edge_weight: float = 0.0,
        label_smoothing: float = 0.1,
        err_weight: float = 0.4,
    ):
        super().__init__()

        # ── Load pretrained encoder ──────────────────────────────────
        self.bert_config = ModernBertConfig.from_pretrained(model_name)
        # Force sdpa attention (not flash_attention_2) for compatibility
        self.bert_config._attn_implementation = "sdpa"
        self.encoder = ModernBertModel.from_pretrained(
            model_name, config=self.bert_config
        )
        self.hidden_size = self.bert_config.hidden_size  # 768

        # ── Initialize Graph-RoPE ────────────────────────────────────
        if rope_config is None:
            rope_config = GraphRoPEConfig(
                d_head=self.bert_config.hidden_size
                // self.bert_config.num_attention_heads,
            )
        self.graph_rope = GraphRoPEEmbedding(rope_config)

        # ── Replace RoPE in all attention layers ─────────────────────
        self._replace_rope()

        # ── Task heads ───────────────────────────────────────────────
        # MLM head: predict masked tokens
        self.mlm_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.bert_config.vocab_size),
        )

        # TFR head: predict exact 6D graph coordinates (Regression)
        self.node_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 6), # Output dim = 6 (f, g_k)
            nn.Sigmoid(),
        )

        # Inter-node relation heads (always created, weight-gated)
        # ERR head: predict log1p(R_eff) from [h_A; h_B; h_A⊙h_B]
        self.err_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        # Huber loss is more robust to the extreme outliers in EfRes targets
        self.err_loss_fn = nn.HuberLoss(delta=1.0)

        # Legacy: Edge prediction head (bilinear scoring)
        self.edge_head = nn.Bilinear(
            self.hidden_size, self.hidden_size, 1
        )
        self.edge_loss_fn = nn.BCEWithLogitsLoss()

        # ── Loss weights and config ──────────────────────────────────
        self.mlm_weight = mlm_weight
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.err_weight = err_weight
        self.label_smoothing = label_smoothing

        # ── Loss functions ───────────────────────────────────────────
        self.mlm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        # Regression Loss for TFR (MSE)
        self.node_loss_fn = nn.MSELoss(reduction='none')

    def _replace_rope(self):
        """
        Replace standard ModernBertRotaryEmbedding with GraphRoPEWrapper
        in every attention layer.

        The wrapper mimics the original interface:
            cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        but returns our pre-computed Graph-RoPE cos/sin.
        """
        self._rope_wrappers = nn.ModuleList()
        for layer in self.encoder.layers:
            attn = layer.attn
            wrapper = GraphRoPEWrapper()
            attn.rotary_emb = wrapper
            self._rope_wrappers.append(wrapper)

    def _set_rope_for_forward(
        self,
        graph_features: torch.Tensor,
    ):
        """
        Compute Graph-RoPE cos/sin and inject into all attention layers.

        Args:
            graph_features: [B, T, 6]
        """
        cos, sin = self.graph_rope(graph_features)
        for wrapper in self._rope_wrappers:
            wrapper.set_cos_sin(cos, sin)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_features: torch.Tensor,
        mlm_labels: torch.Tensor = None,
        node_feature_targets: torch.Tensor = None,
        node_target_mask: torch.Tensor = None,
        edge_pairs: torch.Tensor = None,
        edge_labels: torch.Tensor = None,
        err_pairs: torch.Tensor = None,
        err_targets: torch.Tensor = None,
        **kwargs
    ) -> dict:
        """
        Forward pass with multi-task loss computation.

        Args:
            input_ids:            [B, T] masked token IDs
            attention_mask:       [B, T] attention mask
            graph_features:       [B, T, 6] per-token graph features
            mlm_labels:           [B, T] MLM targets (-100 for non-masked)
            node_feature_targets: [B, T, 6] TFR targets (6D constant)
            node_target_mask:     [B, T] 1.0 where masked, 0.0 otherwise
            edge_pairs:           [B, n_pairs, 2] token pair indices (edge mode)
            edge_labels:          [B, n_pairs] edge labels 0/1 (edge mode)
            err_pairs:            [B, n_pairs, 2] token pair indices (ERR mode)
            err_targets:          [B, n_pairs] log1p(R_eff) values (ERR mode)

        Returns:
            dict with losses and logits
        """
        B, T = input_ids.shape

        # ── Set Graph-RoPE cos/sin for all attention layers ──────────
        self._set_rope_for_forward(graph_features)

        # ── Encode ───────────────────────────────────────────────────
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = encoder_output.last_hidden_state  # [B, T, 768]

        result = {"hidden_states": hidden}
        total_loss = torch.tensor(0.0, device=hidden.device)

        # ── MLM loss ─────────────────────────────────────────────────
        if mlm_labels is not None:
            mlm_logits = self.mlm_head(hidden)  # [B, T, vocab]
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.view(-1, self.bert_config.vocab_size),
                mlm_labels.view(-1),
            )
            total_loss = total_loss + self.mlm_weight * mlm_loss
            result["mlm_loss"] = mlm_loss
            result["mlm_logits"] = mlm_logits

            # MLM accuracy (on masked positions only)
            mask = mlm_labels != -100
            if mask.any():
                preds = mlm_logits[mask].argmax(dim=-1)
                result["mlm_acc"] = (
                    preds == mlm_labels[mask]
                ).float().mean()

        # ── Node feature regression loss (TFR) ────────────────────────
        if node_feature_targets is not None:
            node_preds = self.node_head(hidden)  # [B, T, 6]
            mse = self.node_loss_fn(node_preds, node_feature_targets)  # [B, T, 6]

            if node_target_mask is not None:
                 mask_3d = node_target_mask.unsqueeze(-1)  # [B, T, 1]
                 mse = mse * mask_3d
                 num_masked = mask_3d.sum() + 1e-8
                 node_loss = mse.sum() / num_masked
                 
                 # Debug telemetry if loss is anomalous
                 if node_loss > 10.0:
                     print(f"\n[DEBUG TFR] node_loss={node_loss.item():.4f} | num_masked={num_masked.item():.2f}")
                     print(f"[DEBUG TFR] preds range: [{node_preds.min().item():.4f}, {node_preds.max().item():.4f}]")
                     print(f"[DEBUG TFR] targets range: [{node_feature_targets.min().item():.4f}, {node_feature_targets.max().item():.4f}]")
                     print(f"[DEBUG TFR] mse.sum()={mse.sum().item():.4f}")
            else:
                 node_loss = mse.mean()

            total_loss = total_loss + self.node_weight * node_loss
            result["node_loss"] = node_loss
            result["node_mse"] = node_loss.detach()

        # ── ERR: Effective Resistance Regression ─────────────────────
        if self.err_weight > 0 and err_pairs is not None and err_targets is not None:
            idx_i = err_pairs[:, :, 0]  # [B, n_pairs]
            idx_j = err_pairs[:, :, 1]  # [B, n_pairs]

            h_i = hidden.gather(
                1, idx_i.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            )  # [B, n_pairs, H]
            h_j = hidden.gather(
                1, idx_j.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            )  # [B, n_pairs, H]

            # Feature vector: [h_A; h_B; h_A ⊙ h_B]
            err_features = torch.cat([
                h_i, h_j, h_i * h_j
            ], dim=-1)  # [B, n_pairs, 3H]

            err_pred = self.err_head(err_features).squeeze(-1)  # [B, n_pairs]
            err_loss = self.err_loss_fn(err_pred, err_targets)

            total_loss = total_loss + self.err_weight * err_loss
            result["err_loss"] = err_loss
            result["err_mse"] = err_loss.detach()

        # ── Edge prediction loss ─────────────────────────────────────
        if self.edge_weight > 0 and edge_pairs is not None and edge_labels is not None:
            idx_i = edge_pairs[:, :, 0]
            idx_j = edge_pairs[:, :, 1]

            h_i = hidden.gather(
                1, idx_i.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            )
            h_j = hidden.gather(
                1, idx_j.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            )

            edge_logits = self.edge_head(h_i, h_j).squeeze(-1)
            edge_loss = self.edge_loss_fn(edge_logits, edge_labels.float())
            total_loss = total_loss + self.edge_weight * edge_loss
            result["edge_loss"] = edge_loss

            edge_preds = (edge_logits > 0).long()
            result["edge_acc"] = (
                edge_preds == edge_labels
            ).float().mean()

        result["loss"] = total_loss
        return result

    def get_param_groups(
        self,
        encoder_lr: float = 2e-5,
        rope_lr: float = 1e-4,
        head_lr: float = 5e-5,
        weight_decay: float = 0.01,
    ):
        """
        Differential learning rates for optimizer.
        """
        # 1. Pretrained Encoder (ModernBERT)
        encoder_params = []
        for name, param in self.encoder.named_parameters():
            if "rotary_emb" not in name:
                encoder_params.append(param)
        
        # 2. Graph-RoPE Projection params (Learnable)
        rope_params = list(self.graph_rope.parameters())
        
        # 3. Prediction Heads
        head_params = (
            list(self.mlm_head.parameters()) +
            list(self.node_head.parameters()) +
            list(self.err_head.parameters()) +
            list(self.edge_head.parameters())
        )

        return [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": rope_params, "lr": rope_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
