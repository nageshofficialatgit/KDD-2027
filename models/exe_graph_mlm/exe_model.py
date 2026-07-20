#!/usr/bin/env python3
"""
Exe-MLM Model Architecture.
Wraps `ModernBertForMaskedLM` directly to preserve the 5T-token tied embedding MLM head.
Injects Graph-RoPE structural encodings.
"""

import torch
import torch.nn as nn
from transformers import (
    ModernBertForMaskedLM,
    ModernBertConfig,
)
import sys
import os

# Link to existing wire projection representations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wire.wire_rope import WIREConfig, WIREEmbedding

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

class ExeGraphMLM(nn.Module):
    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        wire_config=None,
        use_pretrained_mlm_head=False, # Ablation toggle: by default, we use a custom newly initialized head
    ):
        super().__init__()
        self.bert_config = ModernBertConfig.from_pretrained(model_name)
        self.bert_config._attn_implementation = "sdpa"
        self.hidden_size = self.bert_config.hidden_size
        
        # Load the Masked LM model
        self.encoder_lm = ModernBertForMaskedLM.from_pretrained(model_name, config=self.bert_config)
        self.encoder = self.encoder_lm.model
        
        self.use_pretrained_mlm_head = use_pretrained_mlm_head
        if not self.use_pretrained_mlm_head:
            # Main Architecture: Recreate random linear head (same as WIRE)
            self.encoder_lm.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), 
                nn.GELU(), 
                nn.LayerNorm(self.hidden_size)
            )
            self.encoder_lm.decoder = nn.Linear(self.hidden_size, self.bert_config.vocab_size)

        if wire_config is None:
            wire_config = WIREConfig(d_head=self.hidden_size // self.bert_config.num_attention_heads)
        self.wire_rope = WIREEmbedding(wire_config)
        self._replace_rope()

    def _replace_rope(self):
        self._rope_wrappers = nn.ModuleList()
        for layer in self.encoder.layers:
            wrapper = WIREWrapper()
            layer.attn.rotary_emb = wrapper
            self._rope_wrappers.append(wrapper)

    def forward(self, input_ids, attention_mask, graph_features, mlm_labels=None, **kwargs):
        # 1. Structural coordinates -> Cos/Sin matrices
        cos, sin = self.wire_rope(graph_features)
        for w in self._rope_wrappers: 
            w.set_cos_sin(cos, sin)

        # 2. Base Encoder pass
        outputs = self.encoder_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels if mlm_labels is not None else None,
            output_hidden_states=True
        )
        
        hidden = outputs.hidden_states[-1]
        total_loss = torch.tensor(0.0, device=hidden.device)
        result = {"hidden_states": hidden}
        
        if mlm_labels is not None:
            total_loss += outputs.loss
            result["mlm_loss"] = outputs.loss
            logits = outputs.logits
            mask = mlm_labels != -100
            if mask.any(): 
                result["mlm_acc"] = (logits[mask].argmax(-1) == mlm_labels[mask]).float().mean()

        result["loss"] = total_loss
        return result

    def get_param_groups(self, encoder_lr=2e-5, wire_lr=1e-4, head_lr=5e-5, weight_decay=0.01):
        # We also need to add decoder if doing custom head
        head_params = list(self.encoder_lm.head.parameters())
        if hasattr(self.encoder_lm, "decoder"):
            head_params += list(self.encoder_lm.decoder.parameters())
            
        return [
            {"params": [p for n, p in self.encoder_lm.named_parameters() if "rotary_emb" not in n and "head" not in n and "decoder" not in n], "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": list(self.wire_rope.parameters()), "lr": wire_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
