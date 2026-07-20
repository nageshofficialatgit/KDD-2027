#!/usr/bin/env python3
import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig, RobertaModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import sys

# Framework imports
ROOT_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from evaluation.benchmarks.tier1_detection.dataset_loader import SupervisedVulnerabilityDataset, get_hash_splits

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    acc = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    
    auc = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            auc.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
        else:
            auc.append(np.nan)
            
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "accuracy": acc,
        "auc": auc,
        "macro_f1": np.mean(f1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["wire", "modernbert", "codebert", "graphcodebert"])
    parser.add_argument("--model_name", type=str, required=True, help="Specific variant name (e.g., e1_err, wire_abl)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    
    EVAL_DIR = os.path.join(ROOT_DIR, "Eval")
    metrics_dir = os.path.join(EVAL_DIR, "metrics", args.model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"--- Starting Evaluation for {args.model_name} ({args.model_type}) ---")

    # 1. Setup Tokenizer & Model
    max_len = 512 if args.model_type in ["codebert", "graphcodebert"] else 2048
    
    if args.model_type == "wire":
        base_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        config = WIREConfig(d_head=64)
        model = WIREModernBERT(base_name, wire_config=config)
        
        if args.checkpoint:
            print(f"Loading WIRE checkpoint from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd, strict=False)
            
        class WIREClassifier(nn.Module):
            def __init__(self, backbone, num_labels=8):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(768, num_labels)
            def forward(self, **batch):
                labels = batch.pop("labels", None)
                outputs = self.backbone(**batch)
                hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
                if hidden is None:
                    hidden = next(iter(outputs.values())) if isinstance(outputs, dict) else outputs.last_hidden_state
                pooled = torch.max(hidden, dim=1)[0]
                logits = self.classifier(pooled)
                loss = nn.BCEWithLogitsLoss()(logits, labels) if labels is not None else None
                return {"loss": loss, "logits": logits}
        model = WIREClassifier(model).to(device)

    else:
        name_map = {
            "modernbert": "answerdotai/ModernBERT-base",
            "codebert": "microsoft/codebert-base",
            "graphcodebert": "microsoft/graphcodebert-base"
        }
        name = name_map[args.model_type]
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=8).to(device)

    # 2. Datasets
    LABELS_PATH = os.path.join(ROOT_DIR, "Implementation", "evaluation", "registry", "labels.jsonl")
    NPZ_DIR = os.path.join(ROOT_DIR, "Implementation", "evaluation", "cache", "graphs", "eval_shards") if args.model_type == "wire" else None
    
    # Note: GraphCodeBERT data flow loading requires a custom wrapper around the dataset
    # For now, we use the text-only approach until the prep script is fully integrated
    full_ds = SupervisedVulnerabilityDataset(LABELS_PATH, tokenizer, max_len=max_len, npz_dir=NPZ_DIR)
    train_idx, val_idx, test_idx = get_hash_splits(LABELS_PATH, seed=args.seed)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size)

    # 3. Training Loop
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val_f1 = 0
    best_test_metrics = None
    
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs["loss"].backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_true, val_probs = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_true.append(batch["labels"].cpu().numpy())
                val_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
        
        metrics = compute_metrics(np.concatenate(val_true), np.concatenate(val_probs))
        print(f"Epoch {epoch} Val Macro-F1: {metrics['macro_f1']:.4f}")
        
        # Test 
        if metrics["macro_f1"] > best_val_f1 or epoch == 0:
            best_val_f1 = metrics["macro_f1"]
            print(f"New best validation Macro-F1: {best_val_f1:.4f}. Saving model weights...")
            torch.save(model.state_dict(), os.path.join(metrics_dir, "best.pt"))
            
            test_true, test_probs = [], []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch} Test"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    test_true.append(batch["labels"].cpu().numpy())
                    test_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            best_test_metrics = compute_metrics(np.concatenate(test_true), np.concatenate(test_probs))

    print(f"\nFinal Test Macro-F1: {best_test_metrics['macro_f1']:.4f}")
    
    # 4. Save Results
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump({
            "model_name": args.model_name,
            "model_type": args.model_type,
            "test_metrics": best_test_metrics
        }, f, indent=2)

if __name__ == "__main__":
    main()
