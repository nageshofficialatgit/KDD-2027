import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import argparse
import sys
from tqdm import tqdm
import numpy as np

# Setup paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from evaluation.phase2.mil_loader import MILLoader, get_mil_splits
from models.mil_head import VulnerabilityMILHead
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

def train_mil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # 2. Datasets
    label_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_labels.jsonl")
    ]
    annotation_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_annotations.jsonl")
    ]
    
    train_indices, val_indices, test_indices = get_mil_splits(label_files)
    
    full_dataset = MILLoader(
        label_files, annotation_files, tokenizer, 
        max_len=args.max_len, 
        npz_dir=args.npz_dir, 
        df_dir=args.df_dir
    )
    
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_indices), batch_size=args.batch_size)
    test_loader = DataLoader(torch.utils.data.Subset(full_dataset, test_indices), batch_size=1) 
    
    # 3. Initialize Model
    if args.model_type == "wire":
        config = WIREConfig(d_head=64)
        encoder = WIREModernBERT(args.base_model_path, wire_config=config)
    else:
        encoder = AutoModel.from_pretrained(args.base_model_path)
        
    if hasattr(encoder, "config"):
        hidden_dim = encoder.config.hidden_size
    else:
        hidden_dim = getattr(encoder, "hidden_size", 768)
        
    mil_head = VulnerabilityMILHead(hidden_dim, num_classes=8)
    
    encoder.to(device)
    mil_head.to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(mil_head.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    for epoch in range(args.epochs):
        encoder.train()
        mil_head.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            line_sink_mask = batch["line_sink_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass based on model type
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                outputs = encoder(**kwargs)
            else:
                outputs = encoder(input_ids, attention_mask=attention_mask)
            
            # Extract last hidden state
            if isinstance(outputs, dict):
                token_embeddings = outputs.get("last_hidden_state")
                if token_embeddings is None:
                    # Fallback for models that might return pooler output or something else
                    token_embeddings = next(iter(outputs.values()))
            else:
                token_embeddings = outputs[0]
            
            # MIL Head - Use line_sink_mask so only the first token of each line contributes to the bag
            mil_out = mil_head(token_embeddings, mask=line_sink_mask)
            bag_logits = mil_out["bag_logits"]
            
            loss = criterion(bag_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f}")
        
    # Final Localization Evaluation
    evaluate_localization(encoder, mil_head, test_loader, device, args)

def evaluate_localization(encoder, mil_head, test_loader, device, args):
    encoder.eval()
    mil_head.eval()
    
    results = []
    top1_hits = 0
    top3_hits = 0
    total_with_gt = 0
    
    output_vis_dir = os.path.join(PROJECT_ROOT, f"Eval_Phase2/localization_results/{args.model_name}")
    os.makedirs(output_vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Localization"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            line_sink_mask = batch["line_sink_mask"].to(device)
            filename = batch["filename"][0]
            
            annotations = test_loader.dataset.dataset.annotations
            if filename not in annotations:
                continue
                
            total_with_gt += 1
            gt_lines = annotations[filename]
            token_to_line = batch["token_to_line"][0].numpy()
            
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                outputs = encoder(**kwargs)
            else:
                outputs = encoder(input_ids, attention_mask=attention_mask)
                
            if isinstance(outputs, dict):
                token_embeddings = outputs.get("last_hidden_state")
                if token_embeddings is None:
                    token_embeddings = next(iter(outputs.values()))
            else:
                token_embeddings = outputs[0]
                
            # Use line_sink_mask for evaluation to only score the first token of each line
            mil_out = mil_head(token_embeddings, mask=line_sink_mask)
            attn_weights = mil_out["attention_weights"].squeeze(0).cpu().numpy()
            
            # Aggregate token weights to lines
            line_scores = {}
            for i, weight in enumerate(attn_weights):
                line_idx = token_to_line[i]
                if line_idx > 0:
                    line_scores[line_idx] = line_scores.get(line_idx, 0) + weight
                    
            # Sort lines by score
            sorted_lines = sorted(line_scores.items(), key=lambda x: x[1], reverse=True)
            top_lines = [l for l, s in sorted_lines]
            
            # Check hits
            if top_lines and top_lines[0] in gt_lines:
                top1_hits += 1
            if any(l in gt_lines for l in top_lines[:3]):
                top3_hits += 1
                
            # Optional: Save visualization for first 20 samples
            if total_with_gt <= 20:
                save_vis(filename, sorted_lines, gt_lines, output_vis_dir)

    print(f"\n--- Localization Results ---")
    print(f"Top-1 Accuracy: {top1_hits/total_with_gt:.4f}")
    print(f"Top-3 Accuracy: {top3_hits/total_with_gt:.4f}")
    
    report = {
        "model_name": args.model_name,
        "top1_acc": top1_hits/total_with_gt,
        "top3_acc": top3_hits/total_with_gt,
        "total_samples": total_with_gt
    }
    report_dir = os.path.join(PROJECT_ROOT, "Eval_Phase2/reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{args.model_name}_localization.json"), 'w') as f:
        json.dump(report, f, indent=4)

def save_vis(filename, sorted_lines, gt_lines, output_dir):
    vis_path = os.path.join(output_dir, f"{filename}_vis.txt")
    with open(vis_path, 'w') as f:
        f.write(f"Vulnerable Lines (Groundtruth): {gt_lines}\n")
        f.write(f"Top Predicted Lines:\n")
        for i, (line, score) in enumerate(sorted_lines[:10]):
            status = "[CORRECT]" if line in gt_lines else ""
            f.write(f"  {i+1}. Line {line}: {score:.4f} {status}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["wire", "modernbert", "codebert", "graphcodebert"])
    parser.add_argument("--base_model_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--npz_dir", type=str, default=None)
    parser.add_argument("--df_dir", type=str, default=None)
    args = parser.parse_args()
    
    train_mil(args)
