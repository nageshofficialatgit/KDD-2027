#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer
import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

def get_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding="max_length").to(device)
    with torch.no_grad():
        if hasattr(model, "backbone") or "WIRE" in str(type(model)):
             g_feats = torch.zeros((1, 1024, 6), device=device)
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], graph_features=g_feats)
        else:
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
        emb = torch.mean(hidden, dim=1).cpu().numpy().squeeze()
    return emb

def run_invariant_probing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 1. Load Models
    model_vanilla = torch.hub.load('huggingface/pytorch-transformers', 'model', base_name).to(device) if False else None # Mock
    from transformers import AutoModel
    model_vanilla = AutoModel.from_pretrained(base_name).to(device)
    
    config = WIREConfig(d_head=64)
    model_new = WIREModernBERT(base_name, wire_config=config).to(device)
    checkpoint_path = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        model_new.load_state_dict(sd, strict=False)

    # 2. Define Logical Pairs
    pairs = [
        {
            "name": "Reentrancy",
            "safe": "function withdraw(uint _amount) public { require(balances[msg.sender] >= _amount); balances[msg.sender] -= _amount; msg.sender.call.value(_amount)(\"\"); }",
            "buggy": "function withdraw(uint _amount) public { require(balances[msg.sender] >= _amount); msg.sender.call.value(_amount)(\"\"); balances[msg.sender] -= _amount; }"
        },
        {
            "name": "Integer Overflow",
            "safe": "function add(uint a, uint b) public pure returns (uint) { uint c = a + b; require(c >= a); return c; }",
            "buggy": "function add(uint a, uint b) public pure returns (uint) { return a + b; }"
        }
    ]

    print("\n--- Invariant Violation Probing (Zero-Shot) ---")
    for pair in pairs:
        print(f"\nTask: {pair['name']}")
        
        # Vanilla distances
        v_safe = get_embedding(model_vanilla, tokenizer, pair['safe'], device)
        v_buggy = get_embedding(model_vanilla, tokenizer, pair['buggy'], device)
        v_dist = cosine(v_safe, v_buggy)
        
        # New model distances
        n_safe = get_embedding(model_new, tokenizer, pair['safe'], device)
        n_buggy = get_embedding(model_new, tokenizer, pair['buggy'], device)
        n_dist = cosine(n_safe, n_buggy)
        
        print(f"  Vanilla Distance: {v_dist:.6f}")
        print(f"  Exe-MLM v5 Distance: {n_dist:.6f}")
        
        if n_dist > v_dist:
            boost = (n_dist - v_dist) / v_dist * 100
            print(f"  RESULT: Exe-MLM is {boost:.2f}% more sensitive to this logical violation!")
        else:
            print("  RESULT: No significant semantic separation detected.")

if __name__ == "__main__":
    run_invariant_probing()
