#!/usr/bin/env python3
"""
Benchmark 2: Audit Fatigue — False Positive Rate on Clean Code
==============================================================
Tests if Exe-MLM generates fewer false alarms on contracts with NO vulnerability.
A model that is "too sensitive" in the wrong direction raises noise, wasting auditor time.

Method: Zero-shot embedding distance from a "Buggy" anchor on clean contracts.
Metric: False Positive Rate (FPR). Lower = Better.
"""
import torch
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

LABELS_PATH = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
SOURCE_DIR  = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/sources")

# "Buggy" anchor: a reentrancy bug
BUGGY_ANCHOR = """
pragma solidity ^0.4.18;
contract Vulnerable {
    mapping(address => uint) public balances;
    function withdraw() public {
        uint amount = balances[msg.sender];
        (bool success,) = msg.sender.call{value: amount}("");  // External call BEFORE state update
        require(success);
        balances[msg.sender] = 0;  // State update AFTER — REENTRANCY!
    }
}
"""

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

def evaluate_model(model, tokenizer, device, is_wire, clean_labels, threshold=0.15):
    """
    For each clean contract, check if the model falsely fires an alert
    (i.e., embedding is within `threshold` cosine distance of the Buggy anchor).
    """
    anchor_buggy = get_emb(model, tokenizer, BUGGY_ANCHOR, device, is_wire)
    fp = 0
    tn = 0

    for item in tqdm(clean_labels, desc="Evaluating"):
        src = item.get("source_code", "")
        if not src or len(src) < 50:
            continue

        emb = get_emb(model, tokenizer, src[:2000], device, is_wire)
        dist = cosine(emb, anchor_buggy)

        # False positive: clean code flagged as buggy (too close to Buggy anchor)
        if dist < threshold:
            fp += 1
        else:
            tn += 1

    total = fp + tn
    fpr = fp / total if total > 0 else 0.0
    return {"fp": fp, "tn": tn, "total": total, "fpr": fpr, "precision": 1.0 - fpr}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base)

    # Load ONLY contracts with ALL labels = 0 (completely clean)
    clean_labels = []
    with open(LABELS_PATH) as f:
        for line in f:
            item = json.loads(line)
            labs = item.get("labels", {})
            if all(v == 0 for v in labs.values()):
                clean_labels.append(item)

    print(f"Clean (safe) contracts found: {len(clean_labels)}")
    clean_labels = clean_labels[:150]  # Cap at 150

    results = {}

    # --- Vanilla ModernBERT ---
    print("\n[1/2] Vanilla ModernBERT")
    vanilla = AutoModel.from_pretrained(base).to(device)
    vanilla.eval()
    results["ModernBERT (Vanilla)"] = evaluate_model(vanilla, tokenizer, device, False, clean_labels)
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
    results["Exe-MLM v5"] = evaluate_model(wire, tokenizer, device, True, clean_labels)
    del wire; torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("BENCHMARK 2: AUDIT FATIGUE (False Positive Rate on Clean Code)")
    print("=" * 60)
    print(f"{'Model':<25} {'FPR':>8} {'Precision':>12} {'Total':>7}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['fpr']:>8.2%} {r['precision']:>12.2%} {r['total']:>7}")
    print("=" * 60)

    winner = min(results, key=lambda k: results[k]["fpr"])
    gap = abs(results["Exe-MLM v5"]["fpr"] - results["ModernBERT (Vanilla)"]["fpr"])
    print(f"\nWINNER (lower FPR): {winner}  |  FPR Gap: {gap:.2%}")

    out_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/semantic/b2_audit_fatigue_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
