import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from tqdm import tqdm
import sys

# Framework imports
ROOT_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from evaluation.benchmarks.tier1_detection.dataset_loader import SupervisedVulnerabilityDataset

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "macro_f1": np.mean(f1)
    }

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

def evaluate_model(model, loader, device, name):
    model.eval()
    y_true, y_probs = [], []
    print(f"\nEvaluating {name} on Decompiled Data...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            y_true.append(batch["labels"].cpu().numpy())
            y_probs.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            
    metrics = compute_metrics(np.concatenate(y_true), np.concatenate(y_probs))
    print(f"[{name}] Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset
    DECOMPILED_PATH = os.path.join(ROOT_DIR, "Implementation", "evaluation", "data", "decompiled", "decompiled_labels.jsonl")
    
    base_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # NOTE: Decompiled data has no graph features, so we pass npz_dir=None
    dataset = SupervisedVulnerabilityDataset(DECOMPILED_PATH, tokenizer, max_len=2048, npz_dir=None)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    print(f"Loaded {len(dataset)} decompiled contracts.")
    
    results = {}
    
    # 2. ModernBERT
    mb_ckpt = os.path.join(ROOT_DIR, "Eval", "metrics", "modernbert", "best.pt")
    if os.path.exists(mb_ckpt):
        mb_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=8)
        mb_model.load_state_dict(torch.load(mb_ckpt, map_location="cpu", weights_only=False))
        mb_model = mb_model.to(device)
        results["modernbert"] = evaluate_model(mb_model, loader, device, "ModernBERT (Decompiled)")
        del mb_model
        torch.cuda.empty_cache()
    else:
        print(f"WARNING: ModernBERT checkpoint not found at {mb_ckpt}")
        
    # 3. Exe-MLM v5
    exe_ckpt = os.path.join(ROOT_DIR, "Eval", "metrics", "exe_mlm_v5", "best.pt")
    if os.path.exists(exe_ckpt):
        config = WIREConfig(d_head=64)
        backbone = WIREModernBERT(base_name, wire_config=config)
        exe_model = WIREClassifier(backbone, num_labels=8)
        exe_model.load_state_dict(torch.load(exe_ckpt, map_location="cpu", weights_only=False))
        exe_model = exe_model.to(device)
        results["exe_mlm_v5"] = evaluate_model(exe_model, loader, device, "Exe-MLM v5 (Decompiled)")
        del exe_model
        torch.cuda.empty_cache()
    else:
        print(f"WARNING: Exe-MLM checkpoint not found at {exe_ckpt}")
        
    # 4. Save results
    OUT_FILE = os.path.join(ROOT_DIR, "Eval", "reports", "decompiled_robustness_results.json")
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_FILE}")

if __name__ == "__main__":
    main()
