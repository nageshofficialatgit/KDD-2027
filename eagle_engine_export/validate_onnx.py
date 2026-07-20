#!/usr/bin/env python3
import os
import torch
import numpy as np
import onnxruntime as ort

from export_onnx import EAGLEExportWrapper, WIREModernBERT, WIREConfig, LLVT

def main():
    checkpoint_path = "./Eval_Phase2/checkpoints/exe_mlm_v5_llvt_text_only/best.pt"
    onnx_path = os.path.join(os.path.dirname(__file__), "eagle.onnx")
    
    # 1. Load PyTorch model
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = WIREModernBERT(
        model_name="answerdotai/ModernBERT-base",
        wire_config=WIREConfig(d_head=64, n_features=6)
    )
    enc_sd = {k: v for k, v in sd["encoder"].items()}
    encoder.load_state_dict(enc_sd, strict=False)
    llvt = LLVT(hidden_dim=768)
    llvt.load_state_dict(sd["llvt"], strict=False)
    model = EAGLEExportWrapper(encoder, llvt)
    model.eval()
    
    # Copy weights for monkey-patched norm
    model.llvt.line_pooler.norm.weight.data = sd["llvt"]["line_pooler.norm.weight"]
    model.llvt.line_pooler.norm.bias.data = sd["llvt"]["line_pooler.norm.bias"]
    
    # 2. Generate random inputs
    # Let's test with a different sequence length to ensure dynamic axes work
    B, T = 1, 1024
    input_ids = torch.randint(0, 50000, (B, T), dtype=torch.long)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    token_to_line = torch.randint(0, 50, (B, T), dtype=torch.long)
    token_to_line[0, 0] = 0
    
    # 3. PyTorch outputs
    with torch.no_grad():
        pt_line_scores, pt_bag_logits, pt_contrastive_emb, pt_line_mask = model(input_ids, attention_mask, token_to_line)
    
    # 4. ONNX outputs
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_to_line": token_to_line.numpy()
    }
    ort_outs = session.run(None, ort_inputs)
    ort_line_scores, ort_bag_logits, ort_contrastive_emb, ort_line_mask = ort_outs
    
    # 5. Compare
    print("Testing PyTorch vs ONNX Output Match:")
    def compare(name, pt, ort_t):
        pt_np = pt.numpy()
        diff = np.abs(pt_np - ort_t).max()
        match = diff < 1e-4
        print(f"[{'PASS' if match else 'FAIL'}] {name:15s} Max Diff: {diff:.6f}")
    
    compare("line_scores", pt_line_scores, ort_line_scores)
    compare("bag_logits", pt_bag_logits, ort_bag_logits)
    compare("contrastive_emb", pt_contrastive_emb, ort_contrastive_emb)
    compare("line_mask", pt_line_mask, ort_line_mask)

if __name__ == "__main__":
    main()
