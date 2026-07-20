#!/usr/bin/env python3
"""
Benchmark 3: Zero-Day Generalization — Novel Pattern Recognition
================================================================
Tests if Exe-MLM can transfer knowledge from one vulnerability type (Reentrancy)
to a completely different one (Unchecked Low-Level Calls) using pure logical reasoning.

A model that only memorizes patterns will fail. One that understands "unsafe external
interactions" will generalize.

Method: Train a minimal linear probe on Reentrancy samples, test on Unchecked_Low_Level_Calls.
Metric: Cross-type F1. Higher = Better semantic generalization.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

LABELS_PATH = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
SOURCE_DIR  = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/sources")

def get_emb(model, tokenizer, text, device, is_wire=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding="max_length").to(device)
    with torch.no_grad():
        if is_wire:
            g = torch.zeros((1, 512, 6), device=device)
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"], graph_features=g)
        else:
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        hidden = out.get("hidden_states", out.get("last_hidden_state")) if isinstance(out, dict) else out.last_hidden_state
        return torch.mean(hidden, dim=1).cpu().numpy().squeeze()

def load_source(contract_id):
    for ext in [".sol", ".txt", ""]:
        p = os.path.join(SOURCE_DIR, contract_id + ext)
        if os.path.exists(p):
            with open(p, "r", errors="ignore") as f:
                return f.read()
    return None

def extract_embeddings(model, tokenizer, device, is_wire, items, label_key):
    """Extract embeddings and binary labels for a set of contracts."""
    embs, labels = [], []
    for item in tqdm(items, desc=f"Embedding ({label_key})"):
        src = item.get("source_code", "")
        if not src or len(src) < 50:
            continue
        emb = get_emb(model, tokenizer, src[:2000], device, is_wire)
        label = int(item["labels"].get(label_key, 0))
        embs.append(emb)
        labels.append(label)
    return np.array(embs), np.array(labels)

def train_and_eval_probe(train_embs, train_labels, test_embs, test_labels):
    """Train a linear probe on train set, evaluate on test set."""
    X_train = torch.FloatTensor(train_embs)
    y_train = torch.FloatTensor(train_labels).unsqueeze(1)
    X_test  = torch.FloatTensor(test_embs)

    probe = nn.Linear(X_train.shape[1], 1)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    for _ in range(10):  # 10 quick epochs
        for xb, yb in loader:
            pred = probe(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X_test)) > 0.5).int().numpy().flatten()

    f1 = f1_score(test_labels, preds, zero_division=0)
    return f1

def evaluate_model(model, tokenizer, device, is_wire, reentrancy, unchecked):
    """
    Train probe on Reentrancy → Test on Unchecked_Low_Level_Calls
    This is the Zero-Day cross-type generalization test.
    """
    print("  Extracting Reentrancy embeddings (train)...")
    train_embs, train_labels = extract_embeddings(
        model, tokenizer, device, is_wire, reentrancy, "Reentrancy")

    print("  Extracting Unchecked Calls embeddings (zero-day test)...")
    test_embs, test_labels = extract_embeddings(
        model, tokenizer, device, is_wire, unchecked, "Unchecked_Low_Level_Calls")

    if len(train_embs) == 0 or len(test_embs) == 0:
        return {"f1": 0.0, "train_n": 0, "test_n": 0}

    f1 = train_and_eval_probe(train_embs, train_labels, test_embs, test_labels)
    return {"f1": f1, "train_n": len(train_embs), "test_n": len(test_embs)}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base)

    # Load and split samples
    reentrancy_items, unchecked_items = [], []
    with open(LABELS_PATH) as f:
        for line in f:
            item = json.loads(line)
            labs = item.get("labels", {})
            if labs.get("Reentrancy", 0) == 1 and labs.get("Unchecked_Low_Level_Calls", 0) == 0:
                reentrancy_items.append(item)
            elif labs.get("Unchecked_Low_Level_Calls", 0) == 1 and labs.get("Reentrancy", 0) == 0:
                unchecked_items.append(item)

    # Balance and cap
    reentrancy_items = reentrancy_items[:120]
    unchecked_items  = unchecked_items[:80]
    print(f"Train (Reentrancy): {len(reentrancy_items)}, Zero-Day Test (Unchecked): {len(unchecked_items)}")

    results = {}

    # --- Vanilla ModernBERT ---
    print("\n[1/2] Vanilla ModernBERT")
    vanilla = AutoModel.from_pretrained(base).to(device)
    vanilla.eval()
    results["ModernBERT (Vanilla)"] = evaluate_model(
        vanilla, tokenizer, device, False, reentrancy_items, unchecked_items)
    del vanilla; torch.cuda.empty_cache()

    # --- Exe-MLM v5 ---
    print("\n[2/2] Exe-MLM v5")
    config = WIREConfig(d_head=64)
    wire = WIREModernBERT(base, wire_config=config).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        wire.load_state_dict(sd, strict=False)
    wire.eval()
    results["Exe-MLM v5"] = evaluate_model(
        wire, tokenizer, device, True, reentrancy_items, unchecked_items)
    del wire; torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("BENCHMARK 3: ZERO-DAY GENERALIZATION (Reentrancy → Unchecked)")
    print("=" * 60)
    print(f"{'Model':<25} {'Cross-Type F1':>14} {'Train N':>8} {'Test N':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['f1']:>14.4f} {r['train_n']:>8} {r['test_n']:>8}")
    print("=" * 60)

    winner = max(results, key=lambda k: results[k]["f1"])
    gap = abs(results["Exe-MLM v5"]["f1"] - results["ModernBERT (Vanilla)"]["f1"])
    print(f"\nWINNER: {winner}  |  F1 Gap: {gap:.4f}")

    out_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/semantic/b3_zero_day_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
