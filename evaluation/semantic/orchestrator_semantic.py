#!/usr/bin/env python3
import torch
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

def get_emb(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
    with torch.no_grad():
        if "WIRE" in str(type(model)):
             g_feats = torch.zeros((1, 512, 6), device=device)
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], graph_features=g_feats)
        else:
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        hidden = outputs.get("last_hidden_state", outputs.get("hidden_states")) if isinstance(outputs, dict) else outputs.last_hidden_state
        return torch.mean(hidden, dim=1).cpu().numpy().squeeze()

def run_batch_semantic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODELS = [
        {"name": "codebert", "type": "hf", "path": "microsoft/codebert-base"},
        {"name": "graphcodebert", "type": "hf", "path": "microsoft/graphcodebert-base"},
        {"name": "modernbert", "type": "hf", "path": "answerdotai/ModernBERT-base"},
        {"name": "wire_abl_comment", "type": "wire", "path": "answerdotai/ModernBERT-base", "ckpt": "Eval_Phase2/llvt/checkpoints/wire_abl_comment_llvt/best.pt"},
        {"name": "exe_mlm_v5", "type": "wire", "path": "answerdotai/ModernBERT-base", "ckpt": "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt"}
    ]
    
    # Semantic Pair: Integer Overflow (Safe vs Buggy)
    safe_code = "function add(uint a, uint b) public pure returns (uint) { uint c = a + b; require(c >= a); return c; }"
    buggy_code = "function add(uint a, uint b) public pure returns (uint) { return a + b; }"
    
    # Perturbation Pair (Same logic, different naming)
    orig_code = "function transfer(address to, uint val) public { balances[msg.sender] -= val; balances[to] += val; }"
    pert_code = "function id_01(address id_02, uint id_03) public { id_04[msg.sender] -= id_03; id_04[id_02] += id_03; }"

    results = []
    
    for m_info in MODELS:
        print(f"Evaluating {m_info['name']}...")
        tokenizer = AutoTokenizer.from_pretrained(m_info['path'])
        
        if m_info['type'] == "wire":
            config = WIREConfig(d_head=64)
            model = WIREModernBERT(m_info['path'], wire_config=config).to(device)
            if 'ckpt' in m_info:
                ckpt_p = os.path.join(PROJECT_ROOT, m_info['ckpt'])
                sd = torch.load(ckpt_p, map_location="cpu")
                # Handle nested dicts
                if "model_state_dict" in sd: sd = sd["model_state_dict"]
                if "encoder" in sd: sd = sd["encoder"] # LLVT ckpt has "encoder" and "llvt"
                sd = {k.replace("encoder.", "").replace("backbone.", ""): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
        else:
            model = AutoModel.from_pretrained(m_info['path']).to(device)
        
        model.eval()
        
        # 1. Logic Gap (Invariant Distance)
        e_safe = get_emb(model, tokenizer, safe_code, device)
        e_buggy = get_emb(model, tokenizer, buggy_code, device)
        logic_gap = cosine(e_safe, e_buggy)
        
        # 2. Robustness (Self-Similarity under perturbation)
        e_orig = get_emb(model, tokenizer, orig_code, device)
        e_pert = get_emb(model, tokenizer, pert_code, device)
        robustness = 1 - cosine(e_orig, e_pert)
        
        results.append({
            "model": m_info['name'],
            "logic_gap": float(logic_gap),
            "robustness": float(robustness)
        })
        
    # Print Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Logic Gap (Inv)':<15} | {'Robustness':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} | {r['logic_gap']:<15.6f} | {r['robustness']:<15.4%}")
    print("="*60)

    with open("Implementation/evaluation/semantic/all_models_semantic.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_batch_semantic()
