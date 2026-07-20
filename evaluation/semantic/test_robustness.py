#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

def get_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding="max_length").to(device)
    with torch.no_grad():
        if hasattr(model, "backbone") or "WIRE" in str(type(model)):
             # Create dummy graph features [B, L, 6]
             g_feats = torch.zeros((1, 1024, 6), device=device)
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], graph_features=g_feats)
        else:
             outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # Extract last hidden state
        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
        else:
            hidden = outputs.last_hidden_state
        
        # Global average pooling for the embedding
        # Avoid [CLS] as it might be too focused on pre-training task
        emb = torch.mean(hidden, dim=1).cpu().numpy().squeeze()
    return emb

def run_robustness_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Models
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # Model A: Vanilla ModernBERT
    print("Loading Vanilla ModernBERT...")
    model_vanilla = AutoModel.from_pretrained(base_name).to(device)
    model_vanilla.eval()
    
    # Model B: Exe-MLM v5 (New Strategy)
    print("Loading Exe-MLM v5...")
    config = WIREConfig(d_head=64)
    model_new = WIREModernBERT(base_name, wire_config=config).to(device)
    
    checkpoint_path = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Clean state dict (remove encoder. prefix if present)
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        model_new.load_state_dict(sd, strict=False)
    model_new.eval()

    # 2. Load Data
    original_file = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
    perturbed_file = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/adversarial_labels.jsonl")
    
    originals = []
    with open(original_file, 'r') as f:
        for i, line in enumerate(f):
            originals.append(json.loads(line))
            if i >= 49: break # Match the 50 limit of perturbed
            
    perturbeds = []
    with open(perturbed_file, 'r') as f:
        for line in f:
            perturbeds.append(json.loads(line))

    # 3. Calculate Stability
    results = []
    
    print("\n--- Running Stability Benchmark (Cosine Similarity) ---")
    for i in tqdm(range(len(perturbeds))):
        orig_src = originals[i]['source_code']
        pert_src = perturbeds[i]['source_code']
        
        # Vanilla stability
        emb_v_o = get_embedding(model_vanilla, tokenizer, orig_src, device)
        emb_v_p = get_embedding(model_vanilla, tokenizer, pert_src, device)
        sim_v = 1 - cosine(emb_v_o, emb_v_p)
        
        # New model stability
        emb_n_o = get_embedding(model_new, tokenizer, orig_src, device)
        emb_n_p = get_embedding(model_new, tokenizer, pert_src, device)
        sim_n = 1 - cosine(emb_n_o, emb_n_p)
        
        results.append({
            "id": i,
            "sim_vanilla": sim_v,
            "sim_new": sim_n
        })

    avg_v = np.mean([r['sim_vanilla'] for r in results])
    avg_n = np.mean([r['sim_new'] for r in results])
    
    print("\n=============================================")
    print(" ROBUSTNESS TEST RESULTS (Semantic Stability) ")
    print("=============================================")
    print(f"Vanilla ModernBERT Avg Similarity: {avg_v:.4f}")
    print(f"Exe-MLM v5 (Trajectory) Avg Similarity: {avg_n:.4f}")
    
    improvement = (avg_n - avg_v) / (1 - avg_v) * 100 if avg_n > avg_v else 0
    print(f"Semantic Alignment Improvement: {improvement:.2f}%")
    print("=============================================")
    
    # Save report
    output_path = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/robustness_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "avg_vanilla": avg_v,
            "avg_new": avg_n,
            "improvement_pct": improvement,
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    run_robustness_test()
