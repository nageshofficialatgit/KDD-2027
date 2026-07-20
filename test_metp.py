import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.metp.metp_dataset import GraphRoPEDataset
from models.metp.metp_model import METPModernBERT

def main():
    print("Testing METP Implementation...")
    # These paths are relative to Implementation dir
    npz_dir = "../model/cpg_features"
    bucket_dir = "../Dataset_for_first/cleaned_buckets_no_comments"
    graph_dir = "../model/cpg_graphs"

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    dataset = GraphRoPEDataset(
        npz_dir=npz_dir,
        bucket_dir=bucket_dir,
        tokenizer=tokenizer,
        max_len=1024,
        n_edge_pairs=128,
        split="val",
        graph_dir=graph_dir
    )

    print(f"Dataset size: {len(dataset)}")
    batch = dataset[0]
    print("1. Dataset __getitem__ successful.")
    
    unique, counts = np.unique(batch["edge_labels"].numpy(), return_counts=True)
    print("   Edge labels distribution:", dict(zip(unique, counts)))

    # Initialize model
    model = METPModernBERT(
        model_name="answerdotai/ModernBERT-base",
        mlm_weight=1.0,
        edge_weight=0.5
    )
    model.eval()
    print("2. Model initialized successfully.")

    # Add batch dimension
    batch_tensors = {k: v.unsqueeze(0) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        out = model(**batch_tensors)
        
    print("3. Model forward pass successful.")
    print(f"   Keys returned: {list(out.keys())}")
    print(f"   Total Loss: {out['loss'].item():.4f}")
    if "edge_loss" in out:
        print(f"   Edge Loss: {out['edge_loss'].item():.4f}")
        print(f"   Edge Acc:  {out['edge_acc'].item():.4f}")
    
    print("TEST PASSED.")

if __name__ == "__main__":
    main()
