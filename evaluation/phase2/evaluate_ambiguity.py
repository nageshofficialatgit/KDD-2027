#!/usr/bin/env python3
import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import sys
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from evaluation.phase2.mil_loader import MILLoader
from models.llvt_model import LLVT
from models.wire.wire_rope_model import WIREModernBERT
from models.wire.wire_rope import WIREConfig

def extract_hidden_states(enc_out):
    if hasattr(enc_out, 'last_hidden_state') and enc_out.last_hidden_state is not None:
        return enc_out.last_hidden_state
    if isinstance(enc_out, dict):
        return enc_out.get("hidden_states") or enc_out.get("last_hidden_state")
    if isinstance(enc_out, (tuple, list)):
        return enc_out[0]
    return enc_out

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # Load Ambiguous Dataset
    dataset = MILLoader(
        [args.label_file], [args.annotation_file], tokenizer,
        max_len=args.max_len, npz_dir=args.npz_dir
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Models
    if args.model_type == "wire":
        encoder = WIREModernBERT(args.base_model_path, wire_config=WIREConfig(d_head=64))
    else:
        encoder = AutoModel.from_pretrained(args.base_model_path)
    
    llvt = LLVT(hidden_dim=768, num_gat_layers=1)
    
    # Load Checkpoint
    print(f"Loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    llvt.load_state_dict(ckpt["llvt"])
    
    encoder.to(device)
    llvt.to(device)
    encoder.eval()
    llvt.eval()
    
    results = {
        "localization": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
        "detection": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    }
    
    vuln_types = dataset.vuln_types
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Ambiguity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            bag_labels = batch["labels"].to(device)
            fn = batch["filename"][0]
            
            # 1. Encode
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                enc_out = encoder(input_ids, attention_mask=attention_mask, graph_features=g_feats.to(device) if g_feats is not None else None)
            else:
                enc_out = encoder(input_ids, attention_mask=attention_mask)
            
            tok_emb = extract_hidden_states(enc_out)
            
            # 2. LLVT Forward
            out = llvt(tok_emb, token_to_line, attention_mask)
            
            # --- Detection Evaluation (Binary) ---
            bag_probs = torch.sigmoid(out["bag_logits"]).squeeze(0) # [8]
            is_vulnerable = bag_labels.max() > 0
            pred_vulnerable = bag_probs.max() > 0.5
            
            if is_vulnerable and pred_vulnerable: results["detection"]["tp"] += 1
            elif not is_vulnerable and pred_vulnerable: results["detection"]["fp"] += 1
            elif is_vulnerable and not pred_vulnerable: results["detection"]["fn"] += 1
            else: results["detection"]["tn"] += 1
            
            # --- Localization Evaluation (LLVT) ---
            if fn in dataset.annotations:
                gt = dataset.annotations[fn]
                scores = out["line_scores"].squeeze(0).cpu().numpy()
                mask = out["line_mask"].squeeze(0).cpu().numpy()
                
                line_scores = [(li, float(scores[li])) for li in range(len(scores)) if mask[li] > 0 and li > 0]
                sorted_lines = [l for l, s in sorted(line_scores, key=lambda x: x[1], reverse=True)]
                
                results["localization"]["total"] += 1
                if sorted_lines and sorted_lines[0] in gt: results["localization"]["top1"] += 1
                if any(l in gt for l in sorted_lines[:3]): results["localization"]["top3"] += 1
                if any(l in gt for l in sorted_lines[:5]): results["localization"]["top5"] += 1

    # Print Report
    print("\n" + "="*50)
    print("  AMBIGUITY ROBUSTNESS REPORT")
    print("="*50)
    
    d = results["detection"]
    precision = d["tp"] / (d["tp"] + d["fp"] + 1e-8)
    recall = d["tp"] / (d["tp"] + d["fn"] + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\n[BINARY DETECTION]")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    l = results["localization"]
    total = max(1, l["total"])
    print(f"\n[LLVT LOCALIZATION]")
    print(f"  Top-1 Recall: {l['top1']/total:.4f}")
    print(f"  Top-3 Recall: {l['top3']/total:.4f}")
    print(f"  Top-5 Recall: {l['top5']/total:.4f}")
    
    # Save JSON
    report_path = os.path.join(PROJECT_ROOT, f"Eval_Phase2/reports/ambiguity_{args.model_name}.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", default="wire")
    parser.add_argument("--base_model_path", default="answerdotai/ModernBERT-base")
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()
    run_evaluation(args)
