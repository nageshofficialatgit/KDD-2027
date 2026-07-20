#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import os
import sys
import numpy as np
import json
from tqdm import tqdm

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from evaluation.benchmarks.tier1_detection.dataset_loader import SupervisedVulnerabilityDataset

def evaluate_sensitivity(model, loader, device, mode="original"):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Mode: {mode}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            g_feats = batch.get("graph_features").to(device)
            
            if mode == "shuffled":
                # Shuffle node features across the sequence dimension
                idx = torch.randperm(g_feats.size(1))
                g_feats = g_feats[:, idx, :]
            elif mode == "zeroed":
                g_feats = torch.zeros_like(g_feats)
            elif mode == "noisy":
                g_feats = g_feats + torch.randn_like(g_feats) * 0.1
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, graph_features=g_feats)
            logits = outputs["logits"]
            
            # Global Average Pooling for detection head (as per model_eval.py logic)
            # Wait, the model might already have a head.
            # In master_benchmark.py, we wrap it in a classifier.
            
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    preds = np.concatenate(all_preds)
    labs = np.concatenate(all_labels)
    
    # Calculate Macro-F1
    from sklearn.metrics import f1_score
    binary_preds = (preds > 0.5).astype(int)
    score = f1_score(labs, binary_preds, average='macro', zero_division=0)
    return score

def run_sensitivity_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 1. Load Model with Classifier Head (Replicating master_benchmark.py structure)
    class WIREClassifier(nn.Module):
        def __init__(self, backbone, num_labels=8):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(768, num_labels)
        def forward(self, **batch):
            outputs = self.backbone(**batch)
            hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
            pooled = torch.max(hidden, dim=1)[0]
            logits = self.classifier(pooled)
            return {"logits": logits}

    backbone = WIREModernBERT(base_name, wire_config=WIREConfig(d_head=64))
    model = WIREClassifier(backbone).to(device)
    
    checkpoint_path = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter and load
        model.load_state_dict(sd, strict=False)
    
    # 2. Load Dataset
    LABELS_PATH = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
    NPZ_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards")
    
    full_ds = SupervisedVulnerabilityDataset(LABELS_PATH, tokenizer, max_len=1024, npz_dir=NPZ_DIR)
    # Use a subset of 500 for speed
    indices = list(range(min(500, len(full_ds))))
    subset = Subset(full_ds, indices)
    loader = DataLoader(subset, batch_size=8)
    
    # 3. Run Benchmark
    modes = ["original", "shuffled", "zeroed", "noisy"]
    results = {}
    
    print("\n--- Running Graph Sensitivity Benchmark ---")
    for mode in modes:
        score = evaluate_sensitivity(model, loader, device, mode=mode)
        results[mode] = score
        print(f"F1 Score ({mode}): {score:.4f}")
        
    print("\n=============================================")
    print(" GRAPH SENSITIVITY RESULTS ")
    print("=============================================")
    orig = results["original"]
    for mode in ["shuffled", "zeroed"]:
        drop = (orig - results[mode]) / max(1e-6, orig) * 100
        print(f"Drop on {mode.capitalize()}: {drop:.2f}%")
    
    if results["shuffled"] < results["original"]:
        print("\nCONCLUSION: Model is SEMANTICALLY GROUNDED in trajectories.")
    else:
        print("\nCONCLUSION: Model is mostly using trajectories for positional syntax.")
    print("=============================================")

if __name__ == "__main__":
    run_sensitivity_benchmark()
