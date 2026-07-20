#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import os
import sys
import numpy as np
from tqdm import tqdm

# Project Setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from evaluation.benchmarks.tier1_detection.dataset_loader import SupervisedVulnerabilityDataset

class FlowProbe(nn.Module):
    """
    A lightweight probe to see if the backbone hidden states 
    contain information about the 'Data-Flow Pressure' (Poisson Potential).
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Predicting the Column 0 of graph features
        )
        
    def forward(self, x):
        return self.regressor(x)

def run_deep_semantic_audit():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="answerdotai/ModernBERT-base")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # 1. Load Backbone
    print(f"Loading Backbone: {base_name}...")
    if "ModernBERT" in base_name:
        config = WIREConfig(d_head=64)
        backbone = WIREModernBERT(base_name, wire_config=config).to(device)
    else:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(base_name).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("encoder.", "").replace("backbone.", ""): v for k, v in sd.items()}
        backbone.load_state_dict(sd, strict=False)
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
        
    probe = FlowProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 2. Load Dataset (v0.5 shards)
    LABELS_PATH = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
    NPZ_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards")
    
    dataset = SupervisedVulnerabilityDataset(LABELS_PATH, tokenizer, max_len=512, npz_dir=NPZ_DIR)
    # Use 100 samples for a quick "Extraction Probe"
    indices = list(range(min(100, len(dataset))))
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=True)
    
    # 3. "Semantic Extraction" (Probing Phase)
    print("\n--- Probing Phase: Extracting Flow Knowledge from Backbone ---")
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Target is Column 0 of graph features (Poisson Pressure)
            targets = batch["graph_features"][:, :, 0].unsqueeze(-1).to(device)
            
            # Get frozen backbone states
            with torch.no_grad():
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, 
                                   graph_features=torch.zeros_like(batch["graph_features"]).to(device))
                hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
                
            # Probe prediction
            preds = probe(hidden)
            
            # Calculate loss only on non-padding tokens
            loss = criterion(preds[attention_mask == 1], targets[attention_mask == 1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Probe Loss: {total_loss/len(loader):.6f}")

    # 4. Final Evaluation: "Logical Sensitivity"
    # We check if the probe's error is low. Low error = High Flow Awareness.
    print("\n=============================================")
    print(" DEEP SEMANTIC AUDIT RESULTS ")
    print("=============================================")
    final_mse = total_loss/len(loader)
    flow_awareness = 1.0 / (1.0 + final_mse) # Score between 0 and 1
    
    print(f"Flow Prediction MSE: {final_mse:.6f}")
    print(f"BACKBONE FLOW AWARENESS: {flow_awareness:.4f}")
    
    if flow_awareness > 0.8:
        print("RESULT: Backbone has DEEP semantic understanding of data-flow.")
    elif flow_awareness > 0.5:
        print("RESULT: Backbone has MODERATE semantic understanding.")
    else:
        print("RESULT: Backbone is primarily focused on syntactic patterns.")
    print("=============================================")

if __name__ == "__main__":
    run_deep_semantic_audit()
