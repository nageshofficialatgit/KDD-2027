#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from evaluation.benchmarks.tier1_detection.dataset_loader import SupervisedVulnerabilityDataset

def test_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # Paths
    ADV_LABELS = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/adversarial_labels.jsonl")
    NPZ_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards")
    MODELS = [
        {"name": "ModernBERT (Vanilla)", "type": "modernbert", "ckpt": None},
        {"name": "Exe-MLM v5 (Trajectory)", "type": "wire", "ckpt": "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt"}
    ]
    
    dataset = SupervisedVulnerabilityDataset(ADV_LABELS, tokenizer, max_len=512, npz_dir=NPZ_DIR)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    final_results = {}
    
    for m_info in MODELS:
        print(f"\nEvaluating {m_info['name']}...")
        if m_info['type'] == "wire":
            config = WIREConfig(d_head=64)
            model = WIREModernBERT(base_name, wire_config=config).to(device)
            if m_info['ckpt']:
                ckpt = torch.load(os.path.join(PROJECT_ROOT, m_info['ckpt']), map_location="cpu")
                sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=1).to(device)
            
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                if m_info['type'] == "wire":
                    # Use zeroed graph features for a true backbone-only detection test if needed, 
                    # but here we use the actual adversarial graph features.
                    outputs = model(input_ids=ids, attention_mask=mask, graph_features=batch["graph_features"].to(device))
                else:
                    outputs = model(input_ids=ids, attention_mask=mask)
                
                # Check for pooler_output or hidden state
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    # Fallback to hidden state mean if no classification head
                    hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
                    logits = torch.mean(hidden, dim=1)[:, :1] # Dummy head
                
                preds = (torch.sigmoid(logits) > 0.5).int()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy()[:, 0]) # Check first label
        
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        final_results[m_info['name']] = {"f1": f1, "acc": acc}

    print("\n--- FINAL COMPARISON ---")
    for name, res in final_results.items():
        print(f"{name:<25}: F1={res['f1']:.4f}, Acc={res['acc']:.4f}")

if __name__ == "__main__":
    test_adversarial()
