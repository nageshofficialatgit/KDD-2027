#!/usr/bin/env python3
"""
EAGLE Phase 1: ONNX Export Pipeline
Replaces `scatter_add_` with Boolean Mask MatMul for ONNX compatibility.
Exports a single dynamic-sequence ONNX graph.
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent dir to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from models.llvt_model import LLVT

class ONNXFriendlyLinePooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_embeddings, token_to_line, attention_mask, max_lines=None):
        B, T, H = token_embeddings.shape
        device = token_embeddings.device
        
        if max_lines is None:
            max_lines = token_to_line.max() + 1
            
        valid = (token_to_line > 0) & (attention_mask > 0)
        lines_idx = torch.arange(max_lines, device=device).view(1, max_lines, 1)
        
        # M[b, l, t] = 1 if token_to_line[b, t] == l and valid
        M = (token_to_line.unsqueeze(1) == lines_idx) & valid.unsqueeze(1)
        M = M.float()
        
        # Matrix multiplication pooling
        line_emb = torch.bmm(M, token_embeddings)
        orig_cnt = M.sum(dim=2)
        
        line_cnt = orig_cnt.clamp(min=1)
        line_emb = line_emb / line_cnt.unsqueeze(-1)
        
        line_mask = (orig_cnt > 0).float()
        line_mask[:, 0] = 0
        
        line_emb = self.norm(line_emb)
        return line_emb, line_mask

class EAGLEExportWrapper(nn.Module):
    def __init__(self, encoder, llvt):
        super().__init__()
        self.encoder = encoder
        self.llvt = llvt
        
        # Monkey patch the line pooler for ONNX
        hidden_dim = llvt.line_pooler.norm.weight.shape[0]
        self.llvt.line_pooler = ONNXFriendlyLinePooler(hidden_dim)
        
    def forward(self, input_ids, attention_mask, token_to_line):
        B, T = input_ids.shape
        device = input_ids.device
        
        # 1. Forward through encoder with dummy zeros for graph features (folds cos/sin)
        graph_features = torch.zeros(B, T, 6, device=device)
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, graph_features=graph_features)
        hidden = enc_out["hidden_states"]
        
        # 2. Forward through LLVT
        llvt_out = self.llvt(
            token_embeddings=hidden, 
            token_to_line=token_to_line, 
            attention_mask=attention_mask
        )
        
        line_scores = llvt_out["line_scores"]
        bag_logits = llvt_out["bag_logits"]
        line_mask = llvt_out["line_mask"]
        
        # 3. Explicitly compute contrastive projection for localization guardrails
        line_embeddings = llvt_out["line_embeddings"]
        contrastive_emb = self.llvt.contrastive_proj(line_embeddings)
        
        return line_scores, bag_logits, contrastive_emb, line_mask

def main():
    checkpoint_path = "./Eval_Phase2/checkpoints/exe_mlm_v5_llvt_text_only/best.pt"
    out_onnx_path = os.path.join(os.path.dirname(__file__), "eagle.onnx")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 1. Init Encoder
    print("Initializing WIREModernBERT...")
    encoder = WIREModernBERT(
        model_name="answerdotai/ModernBERT-base",
        wire_config=WIREConfig(d_head=64, n_features=6)
    )
    # The checkpoint state dict has "encoder." prefix in the checkpoint, but our WIREModernBERT 
    # itself might not need it if the wrapper was different. Wait, in checkpoint it is literally "encoder".
    # Let's clean the state dict keys
    enc_sd = {k: v for k, v in sd["encoder"].items()}
    encoder.load_state_dict(enc_sd, strict=False)
    
    # 2. Init LLVT
    print("Initializing LLVT...")
    llvt = LLVT(hidden_dim=768)
    llvt.load_state_dict(sd["llvt"], strict=False)
    
    # 3. Wrap
    model = EAGLEExportWrapper(encoder, llvt)
    model.eval()
    
    # Copy weights for line pooler norm (which we monkey patched)
    model.llvt.line_pooler.norm.weight.data = sd["llvt"]["line_pooler.norm.weight"]
    model.llvt.line_pooler.norm.bias.data = sd["llvt"]["line_pooler.norm.bias"]
    
    print("Preparing dummy inputs for ONNX trace...")
    # B=1, T=256
    dummy_input_ids = torch.randint(0, 50000, (1, 256), dtype=torch.long)
    dummy_attention = torch.ones(1, 256, dtype=torch.long)
    dummy_token_to_line = torch.randint(0, 10, (1, 256), dtype=torch.long)
    dummy_token_to_line[0, 0] = 0 # CLS token mapped to padding line
    
    print(f"Exporting to {out_onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention, dummy_token_to_line),
        out_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_to_line"],
        output_names=["line_scores", "bag_logits", "contrastive_emb", "line_mask"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "token_to_line": {1: "seq_len"},
            "line_scores": {1: "max_lines"},
            "contrastive_emb": {1: "max_lines"},
            "line_mask": {1: "max_lines"}
        }
    )
    
    print("ONNX export complete!")
    print(f"Artifact size: {os.path.getsize(out_onnx_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
