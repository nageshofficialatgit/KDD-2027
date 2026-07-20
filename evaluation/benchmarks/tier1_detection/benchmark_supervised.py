import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import sys

# Framework imports
sys.path.append("./Implementation")
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from dataset_loader import SupervisedVulnerabilityDataset, get_hash_splits

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    
    # Calculate per-class metrics
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    acc = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    
    # Calculate AUC (handle cases with only one class)
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
        except:
            aucs.append(0.5)
            
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "accuracy": acc,
        "auc": aucs,
        "macro_f1": np.mean(f1)
    }

def train_eval(custom_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["wire", "modernbert", "codebert"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--mode", type=str, default="probe", choices=["probe", "finetune"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size for dry-run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    
    if custom_args is not None:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # 1. Setup Tokenizer & Model
    if args.model_type == "wire":
        base_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        config = WIREConfig(d_head=64) # standard for base
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
                loss = None
                if labels is not None:
                    loss = nn.BCEWithLogitsLoss()(logits, labels)
                return {"loss": loss, "logits": logits}
        model = WIREClassifier(model).to(device)

    else:
        name = "answerdotai/ModernBERT-base" if args.model_type == "modernbert" else "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=8).to(device)

    # 2. Freezing for Probing
    if args.mode == "probe":
        print("Freezing backbone for Linear Probing...")
        for name, param in model.named_parameters():
             if "classifier" not in name:
                  param.requires_grad = False

    # 3. Datasets
    LABELS_PATH = "./evaluation/registry/labels.jsonl"
    NPZ_DIR = "./evaluation/cache/graphs/eval_shards" if args.model_type == "wire" else None
    
    max_len = 512 if args.model_type == "codebert" else 2048
    full_ds = SupervisedVulnerabilityDataset(LABELS_PATH, tokenizer, max_len=max_len, npz_dir=NPZ_DIR)
    train_idx, val_idx, test_idx = get_hash_splits(LABELS_PATH, seed=args.seed)
    
    if args.limit > 0:
        train_idx, val_idx, test_idx = train_idx[:args.limit], val_idx[:args.limit], test_idx[:args.limit]

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size)

    # 4. Training
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    results = []
    best_val_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_true = []
        val_probs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_true.append(batch["labels"].cpu().numpy())
                val_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
        
        metrics = compute_metrics(np.concatenate(val_true), np.concatenate(val_probs))
        print(f"Epoch {epoch} Val F1: {metrics['macro_f1']:.4f}")
        results.append({"epoch": epoch, "val_metrics": metrics})
        best_val_f1 = max(best_val_f1, metrics["macro_f1"])

    # 5. Final Test Evaluation
    model.eval()
    test_true = []
    test_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            test_true.append(batch["labels"].cpu().numpy())
            test_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            
    final_metrics = compute_metrics(np.concatenate(test_true), np.concatenate(test_probs))
    print(f"Final Test F1: {final_metrics['macro_f1']:.4f}")
    
    # 6. Save Results
    base_output_dir = f"./evaluation/results/tier1_detection/{args.model_type}_{args.mode}"
    if args.output_suffix:
        output_dir = os.path.join(base_output_dir, args.output_suffix)
    else:
        output_dir = base_output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "final_test": final_metrics, "history": results}, f, indent=2)
    
    return best_val_f1

if __name__ == "__main__":
    train_eval()
