#!/usr/bin/env python3
"""
WIRE-ModernBERT Ablation Model.
Wraps pretrained ModernBERT with standalone WIRE projection-based RoPE.
"""

import torch
import torch.nn as nn
from transformers import (
    ModernBertModel,
    ModernBertConfig,
)

from models.wire.wire_rope import (
    WIREConfig,
    WIREEmbedding,
)

class WIREWrapper(nn.Module):
    """Mimic ModernBERT rotary interface for pre-computed WIRE angles."""
    def __init__(self):
        super().__init__()
        self._cos = None
        self._sin = None

    def set_cos_sin(self, cos, sin):
        self._cos = cos
        self._sin = sin

    def forward(self, x, position_ids=None, **kwargs):
        cos = torch.cat([self._cos, self._cos], dim=-1)
        sin = torch.cat([self._sin, self._sin], dim=-1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class METPModernBERT(nn.Module):
    """METP variant using Structural Projection."""
    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        wire_config=None,
        mlm_weight=1.0,
        edge_weight=0.5,
        label_smoothing=0.1,
    ):
        super().__init__()
        self.bert_config = ModernBertConfig.from_pretrained(model_name)
        self.bert_config._attn_implementation = "sdpa"
        self.encoder = ModernBertModel.from_pretrained(model_name, config=self.bert_config)
        self.config = self.bert_config
        self.hidden_size = self.bert_config.hidden_size

        # ── WIRE Embedding ──
        if wire_config is None:
            wire_config = WIREConfig(d_head=self.hidden_size // self.bert_config.num_attention_heads)
        self.wire_rope = WIREEmbedding(wire_config)
        self._replace_rope()

        # ── Heads ──
        self.mlm_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, self.bert_config.vocab_size))
        
        # METP Head: 4 classes (CFG, DFG, CG, NONE)
        self.edge_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size), 
            nn.LayerNorm(self.hidden_size), 
            nn.GELU(), 
            nn.Linear(self.hidden_size, 4)
        )
        
        self.mlm_weight = mlm_weight
        self.edge_weight = edge_weight
        self.edge_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)

    def _replace_rope(self):
        self._rope_wrappers = nn.ModuleList()
        for layer in self.encoder.layers:
            wrapper = WIREWrapper()
            layer.attn.rotary_emb = wrapper
            self._rope_wrappers.append(wrapper)

    def forward(self, input_ids, attention_mask, graph_features, mlm_labels=None, edge_pairs=None, edge_labels=None, **kwargs):
        cos, sin = self.wire_rope(graph_features)
        for w in self._rope_wrappers: w.set_cos_sin(cos, sin)

        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        result = {"hidden_states": hidden}
        total_loss = torch.tensor(0.0, device=hidden.device)

        if mlm_labels is not None:
            logits = self.mlm_head(hidden)
            loss = self.mlm_loss_fn(logits.view(-1, self.bert_config.vocab_size), mlm_labels.view(-1))
            total_loss += self.mlm_weight * loss
            result["mlm_loss"] = loss
            mask = mlm_labels != -100
            if mask.any(): result["mlm_acc"] = (logits[mask].argmax(-1) == mlm_labels[mask]).float().mean()

        if self.edge_weight > 0 and edge_pairs is not None and edge_labels is not None:
            hi = hidden.gather(1, edge_pairs[:,:,0].unsqueeze(-1).expand(-1,-1,self.hidden_size))
            hj = hidden.gather(1, edge_pairs[:,:,1].unsqueeze(-1).expand(-1,-1,self.hidden_size))
            
            # Predict edge types based on concatenated node states
            edge_logits = self.edge_head(torch.cat([hi, hj, hi*hj], -1)) # [B, n_pairs, 4]
            loss = self.edge_loss_fn(edge_logits.view(-1, 4), edge_labels.view(-1))
            
            total_loss += self.edge_weight * loss
            result["edge_loss"] = loss
            
            edge_preds = edge_logits.argmax(-1)
            result["edge_acc"] = (edge_preds == edge_labels).float().mean()

        result["loss"] = total_loss
        return result

    def get_param_groups(self, encoder_lr=2e-5, wire_lr=1e-4, head_lr=5e-5, weight_decay=0.01):
        return [
            {"params": [p for n, p in self.encoder.named_parameters() if "rotary_emb" not in n], "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": list(self.wire_rope.parameters()), "lr": wire_lr, "weight_decay": weight_decay},
            {"params": list(self.mlm_head.parameters()) + list(self.edge_head.parameters()), "lr": head_lr, "weight_decay": weight_decay},
        ]
