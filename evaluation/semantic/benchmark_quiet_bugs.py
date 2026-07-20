#!/usr/bin/env python3
"""
Benchmark 1: False Negative Reduction — "Quiet Bug Detection"
=============================================================
Tests if Exe-MLM catches subtle/quiet bugs (Integer Overflow, Unchecked Calls)
that look syntactically safe, where vanilla models fail silently.

Method: Zero-shot embedding cosine distance from a "Buggy" anchor.
Metric: False Negative Rate (FNR). Lower = Better.
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
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/sources")

# Quiet Bug anchor: a known Integer Overflow (no require guard)
QUIET_BUG_ANCHOR = """
pragma solidity ^0.4.18;
contract Overflow {
    uint public balance = 1;
    function add(uint value) public {
        balance += value;  // SILENT OVERFLOW - no check
    }
}
"""

# Safe anchor: same logic but protected
SAFE_ANCHOR = """
pragma solidity ^0.6.0;
import "@openzeppelin/contracts/math/SafeMath.sol";
contract SafeAdd {
    using SafeMath for uint256;
    uint public balance = 1;
    function add(uint value) public {
        balance = balance.add(value);  // SAFE - reverts on overflow
    }
}
"""

QUIET_VULN_TYPES = ["Integer_Overflow_Underflow", "Unchecked_Low_Level_Calls"]

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
    """Try to load source code from registry."""
    for ext in [".sol", ".txt", ""]:
        p = os.path.join(SOURCE_DIR, contract_id + ext)
        if os.path.exists(p):
            with open(p, "r", errors="ignore") as f:
                return f.read()
    return None

def evaluate_model(model, tokenizer, device, is_wire, labels, threshold=0.05):
    """
    For each contract with a quiet vuln label, check if the model's embedding
    is close to the Buggy anchor (correct detection) or the Safe anchor (FN).
    """
    anchor_buggy = get_emb(model, tokenizer, QUIET_BUG_ANCHOR, device, is_wire)
    anchor_safe  = get_emb(model, tokenizer, SAFE_ANCHOR,      device, is_wire)

    tp = 0  # Correctly flagged as buggy
    fn = 0  # Missed — treated as safe

    for item in tqdm(labels, desc="Evaluating"):
        src = item.get("source_code", "")
        if not src or len(src) < 50:
            continue

        emb = get_emb(model, tokenizer, src[:2000], device, is_wire)

        dist_buggy = cosine(emb, anchor_buggy)
        dist_safe  = cosine(emb, anchor_safe)

        # Correct detection: closer to Buggy anchor than Safe anchor
        if dist_buggy < dist_safe:
            tp += 1
        else:
            fn += 1

    total = tp + fn
    fnr = fn / total if total > 0 else 1.0
    return {"tp": tp, "fn": fn, "total": total, "fnr": fnr, "detection_rate": 1.0 - fnr}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base)

    # Load only "quiet bug" samples
    quiet_labels = []
    with open(LABELS_PATH) as f:
        for line in f:
            item = json.loads(line)
            labs = item.get("labels", {})
            is_quiet = any(labs.get(vt, 0) == 1 for vt in QUIET_VULN_TYPES)
            # Exclude Reentrancy to keep these "subtle" (no obvious external call)
            is_noisy = labs.get("Reentrancy", 0) == 1
            if is_quiet and not is_noisy:
                quiet_labels.append(item)

    print(f"Quiet bug samples: {len(quiet_labels)}")
    # Use up to 150 samples
    quiet_labels = quiet_labels[:150]

    results = {}

    # --- Vanilla ModernBERT ---
    print("\n[1/2] Vanilla ModernBERT")
    vanilla = AutoModel.from_pretrained(base).to(device)
    vanilla.eval()
    results["ModernBERT (Vanilla)"] = evaluate_model(vanilla, tokenizer, device, False, quiet_labels)
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
    results["Exe-MLM v5"] = evaluate_model(wire, tokenizer, device, True, quiet_labels)
    del wire; torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 60)
    print("BENCHMARK 1: QUIET BUG DETECTION (False Negative Rate)")
    print("=" * 60)
    print(f"{'Model':<25} {'FNR':>8} {'Detection Rate':>16} {'Total':>7}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25} {r['fnr']:>8.2%} {r['detection_rate']:>16.2%} {r['total']:>7}")
    print("=" * 60)

    winner = min(results, key=lambda k: results[k]["fnr"])
    gap = abs(results["Exe-MLM v5"]["fnr"] - results["ModernBERT (Vanilla)"]["fnr"])
    print(f"\nWINNER: {winner}  |  FNR Gap: {gap:.2%}")

    out_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/semantic/b1_quiet_bug_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
