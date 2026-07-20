#!/usr/bin/env python3
"""
Compare Pre-trained Exe-MLM vs Untrained Baseline.

Trains two identical WIRE+LLVT pipelines on the vulnerability localization task:
  1. "pretrained" — Encoder initialised from the Exe-MLM checkpoint (our model)
  2. "baseline"   — Encoder initialised from vanilla ModernBERT (no pre-training)

Both use the same LLVT head (randomly initialised), same data splits,
same hyper-parameters and same number of fine-tuning epochs.

After training, runs identical test-set evaluation and writes a JSON report
with side-by-side metrics.
"""

import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import sys
import random
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from evaluation.phase2.mil_loader import MILLoader
from models.llvt_model import LLVT
from models.wire.wire_rope_model import WIREModernBERT
from models.wire.wire_rope import WIREConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_hidden_states(enc_out):
    if hasattr(enc_out, 'last_hidden_state') and enc_out.last_hidden_state is not None:
        return enc_out.last_hidden_state
    if isinstance(enc_out, dict):
        if "hidden_states" in enc_out and enc_out["hidden_states"] is not None:
            return enc_out["hidden_states"]
        if "last_hidden_state" in enc_out and enc_out["last_hidden_state"] is not None:
            return enc_out["last_hidden_state"]
        for v in enc_out.values():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                return v
    if isinstance(enc_out, (tuple, list)):
        return enc_out[0]
    raise ValueError(f"Cannot extract hidden states from {type(enc_out)}")


def get_splits(label_files, annotation_files, seed=42):
    """Deterministic 70/15/15 split with curated files represented in test."""
    rng = random.Random(seed)
    all_data = []
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))

    curated_filenames = set()
    for af in annotation_files:
        if not os.path.exists(af):
            continue
        with open(af, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get("dataset") == "SBCurated":
                    curated_filenames.add(d["filename"])

    curated_idx = [i for i, d in enumerate(all_data) if d["filename"] in curated_filenames]
    other_idx = [i for i, d in enumerate(all_data) if d["filename"] not in curated_filenames]
    rng.shuffle(curated_idx)
    rng.shuffle(other_idx)

    def _split(lst):
        n = len(lst)
        nv = max(1, int(n * 0.15))
        nt = max(1, int(n * 0.15))
        return lst[nv + nt:], lst[:nv], lst[nv:nv + nt]

    ct, cv, ce = _split(curated_idx)
    ot, ov, oe = _split(other_idx)
    train = ct + ot
    val = cv + ov
    test = ce + oe
    rng.shuffle(train)
    return train, val, test


def build_line_labels(fn, annotations, curated_files, L):
    labels = torch.zeros(L)
    weights = torch.ones(L)
    is_curated = fn in curated_files
    if fn in annotations:
        conf = 1.0 if is_curated else 0.3
        weights[:] = conf
        for li in annotations[fn]:
            if 0 < li < L:
                labels[li] = 1.0
                weights[li] = 1.0
        if not is_curated:
            labels = labels * 0.85 + 0.05 * (1 - labels) * (labels == 0).float()
    else:
        weights[:] = 0.0
    return labels, weights


# ---------------------------------------------------------------------------
# Training loop (shared by both variants)
# ---------------------------------------------------------------------------

def train_and_evaluate(
    encoder, llvt, train_loader, val_loader, test_loader,
    annotations, curated_files, device, tag, epochs=10, lr=2e-5
):
    """Train encoder+LLVT, evaluate, return results dict."""

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr},
        {"params": llvt.parameters(), "lr": lr * 5},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = 0
    best_state = None

    for epoch in range(epochs):
        encoder.train()
        llvt.train()
        ep_loss = 0
        n = 0

        for batch in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bag_labels = batch["labels"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]

            # Encode
            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if g_feats is not None:
                kwargs["graph_features"] = g_feats.to(device)
            else:
                kwargs["graph_features"] = torch.zeros(
                    (input_ids.size(0), input_ids.size(1), 6), device=device
                )
            enc_out = encoder(**kwargs)
            tok_emb = extract_hidden_states(enc_out)

            B = input_ids.size(0)
            max_line = token_to_line.max().item()
            L = max(max_line + 1, 2)

            all_line_labels = torch.zeros(B, L, device=device)
            all_line_weights = torch.ones(B, L, device=device)
            for b in range(B):
                ll, ww = build_line_labels(filenames[b], annotations, curated_files, L)
                all_line_labels[b] = ll.to(device)
                all_line_weights[b] = ww.to(device)

            out = llvt(
                tok_emb, token_to_line, attention_mask,
                line_labels=all_line_labels,
                line_label_weights=all_line_weights,
                bag_labels=bag_labels,
            )

            loss = out["line_loss"] + 0.3 * out.get("bag_loss", 0) + 0.1 * out.get("contrastive_loss", 0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(llvt.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            n += 1

        avg = ep_loss / max(n, 1)
        print(f"[{tag}] Epoch {epoch+1}/{epochs}  loss={avg:.4f}")

        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            val_met = _validate(encoder, llvt, val_loader, device, annotations, curated_files)
            print(f"[{tag}]   Val Top-5 Recall: {val_met:.4f}")
            if val_met > best_val:
                best_val = val_met
                best_state = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "llvt": copy.deepcopy(llvt.state_dict()),
                }

    # Restore best
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        llvt.load_state_dict(best_state["llvt"])
        print(f"[{tag}] Restored best checkpoint (val_top5={best_val:.4f})")

    # Final evaluation
    results = _evaluate(encoder, llvt, test_loader, device, annotations, curated_files)
    results["best_val_top5"] = best_val
    return results


def _validate(encoder, llvt, loader, device, annotations, curated_files):
    encoder.eval()
    llvt.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]

            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if g_feats is not None:
                kwargs["graph_features"] = g_feats.to(device)
            else:
                kwargs["graph_features"] = torch.zeros(
                    (input_ids.size(0), input_ids.size(1), 6), device=device
                )
            enc_out = encoder(**kwargs)
            tok_emb = extract_hidden_states(enc_out)
            out = llvt(tok_emb, token_to_line, attention_mask)

            for b in range(input_ids.size(0)):
                fn = filenames[b]
                if fn not in annotations:
                    continue
                gt = annotations[fn]
                scores = out["line_scores"][b].cpu().numpy()
                mask = out["line_mask"][b].cpu().numpy()
                ls = [(li, scores[li]) for li in range(len(scores)) if mask[li] > 0 and li > 0]
                top5 = [l for l, s in sorted(ls, key=lambda x: x[1], reverse=True)[:5]]
                total += 1
                if any(l in gt for l in top5):
                    hits += 1
    return hits / max(1, total)


def _evaluate(encoder, llvt, loader, device, annotations, curated_files):
    encoder.eval()
    llvt.eval()

    results = {
        "all":     {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
        "curated": {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
        "scrawld": {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
    }
    detection = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            bag_labels = batch["labels"].to(device)
            fn = batch["filename"][0]

            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if g_feats is not None:
                kwargs["graph_features"] = g_feats.to(device)
            else:
                kwargs["graph_features"] = torch.zeros(
                    (input_ids.size(0), input_ids.size(1), 6), device=device
                )
            enc_out = encoder(**kwargs)
            tok_emb = extract_hidden_states(enc_out)
            out = llvt(tok_emb, token_to_line, attention_mask)

            # --- Detection ---
            bag_probs = torch.sigmoid(out["bag_logits"]).squeeze(0)
            is_vuln = bag_labels.max() > 0
            pred_vuln = bag_probs.max() > 0.5
            if is_vuln and pred_vuln:       detection["tp"] += 1
            elif not is_vuln and pred_vuln: detection["fp"] += 1
            elif is_vuln and not pred_vuln: detection["fn"] += 1
            else:                           detection["tn"] += 1

            # --- Localization ---
            if fn not in annotations:
                continue
            gt = annotations[fn]
            is_curated = fn in curated_files

            scores = out["line_scores"].squeeze(0).cpu().numpy()
            mask = out["line_mask"].squeeze(0).cpu().numpy()
            ls = [(li, float(scores[li])) for li in range(len(scores)) if mask[li] > 0 and li > 0]
            top_lines = [l for l, s in sorted(ls, key=lambda x: x[1], reverse=True)]

            for bucket in ["all", "curated" if is_curated else "scrawld"]:
                r = results[bucket]
                r["total"] += 1
                if top_lines and top_lines[0] in gt:     r["top1"] += 1
                if any(l in gt for l in top_lines[:3]):  r["top3"] += 1
                if any(l in gt for l in top_lines[:5]):  r["top5"] += 1
                if any(l in gt for l in top_lines[:10]): r["top10"] += 1

    # Compute F1
    p = detection["tp"] / (detection["tp"] + detection["fp"] + 1e-8)
    r = detection["tp"] / (detection["tp"] + detection["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return {
        "localization": {
            name: {k: v / max(1, vals["total"]) if k != "total" else v for k, v in vals.items()}
            for name, vals in results.items()
        },
        "detection": {**detection, "precision": p, "recall": r, "f1": f1},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare pre-trained Exe-MLM vs vanilla ModernBERT baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--exe_mlm_ckpt",
        default=os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt"),
        help="Path to the Exe-MLM pre-training checkpoint")
    parser.add_argument("--npz_dir",
        default=os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards"),
        help="Path to graph feature NPZ shards")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # --- Data ---
    label_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_labels.jsonl"),
    ]
    annotation_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_annotations.jsonl"),
    ]

    annotations = {}
    curated_files = set()
    for af in annotation_files:
        if not os.path.exists(af):
            continue
        with open(af, 'r') as f:
            for line in f:
                d = json.loads(line)
                fn = d["filename"]
                ds = d.get("dataset", "ScrawlD")
                if ds == "SBCurated" or fn not in annotations:
                    annotations[fn] = d["vulnerable_lines"]
                if ds == "SBCurated":
                    curated_files.add(fn)
    print(f"Annotations: {len(annotations)} files ({len(curated_files)} curated)")

    dataset = MILLoader(
        label_files, annotation_files, tokenizer,
        max_len=args.max_len, npz_dir=args.npz_dir,
    )
    dataset.annotations = annotations

    train_idx, val_idx, test_idx = get_splits(label_files, annotation_files, seed=args.seed)
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=1)

    # =====================================================================
    # VARIANT A: Pre-trained Exe-MLM  (our model)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  VARIANT A: Pre-trained Exe-MLM + LLVT")
    print("=" * 60)

    seed_everything(args.seed)  # reset so LLVT init is identical
    encoder_A = WIREModernBERT(args.base_model, wire_config=WIREConfig(d_head=64))

    # Load Exe-MLM pre-trained weights into the encoder
    ckpt = torch.load(args.exe_mlm_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    # Map keys: the pre-training checkpoint has "encoder.xxx" keys
    encoder_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            encoder_sd[k.replace("encoder.", "")] = v
    missing, unexpected = encoder_A.encoder.load_state_dict(encoder_sd, strict=False)
    print(f"Loaded Exe-MLM weights: {len(encoder_sd)} keys, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    # Also load wire_rope weights if present
    wire_sd = {k.replace("wire_rope.", ""): v for k, v in sd.items() if k.startswith("wire_rope.")}
    if wire_sd:
        encoder_A.wire_rope.load_state_dict(wire_sd, strict=False)
        print(f"Loaded WIRE-RoPE weights: {len(wire_sd)} keys")

    llvt_A = LLVT(hidden_dim=768, num_gat_layers=1, num_heads=4, dropout=0.1,
                  focal_gamma=2.0, focal_alpha=0.25, contrastive_weight=0.1)

    encoder_A.to(device)
    llvt_A.to(device)

    results_A = train_and_evaluate(
        encoder_A, llvt_A, train_loader, val_loader, test_loader,
        annotations, curated_files, device,
        tag="PRETRAINED", epochs=args.epochs, lr=args.lr,
    )

    # Free memory
    del encoder_A, llvt_A
    torch.cuda.empty_cache()

    # =====================================================================
    # VARIANT B: Vanilla ModernBERT baseline  (untrained / from-scratch)
    # =====================================================================
    print("\n" + "=" * 60)
    print("  VARIANT B: Vanilla ModernBERT (No Pre-training) + LLVT")
    print("=" * 60)

    seed_everything(args.seed)  # identical LLVT init
    encoder_B = WIREModernBERT(args.base_model, wire_config=WIREConfig(d_head=64))
    # NO checkpoint loaded — uses vanilla HuggingFace ModernBERT weights

    llvt_B = LLVT(hidden_dim=768, num_gat_layers=1, num_heads=4, dropout=0.1,
                  focal_gamma=2.0, focal_alpha=0.25, contrastive_weight=0.1)

    encoder_B.to(device)
    llvt_B.to(device)

    results_B = train_and_evaluate(
        encoder_B, llvt_B, train_loader, val_loader, test_loader,
        annotations, curated_files, device,
        tag="BASELINE", epochs=args.epochs, lr=args.lr,
    )

    # =====================================================================
    # Comparison Report
    # =====================================================================
    print("\n" + "=" * 70)
    print("  COMPARISON: Pre-trained Exe-MLM  vs  Vanilla ModernBERT")
    print("=" * 70)

    def _fmt(val):
        return f"{val:.4f}"

    # Detection
    dA = results_A["detection"]
    dB = results_B["detection"]
    print("\n[BINARY DETECTION]")
    print(f"  {'Metric':<15} {'Pretrained':>12} {'Baseline':>12} {'Δ':>10}")
    print(f"  {'-'*49}")
    for m in ["f1", "precision", "recall"]:
        delta = dA[m] - dB[m]
        sign = "+" if delta >= 0 else ""
        print(f"  {m.capitalize():<15} {_fmt(dA[m]):>12} {_fmt(dB[m]):>12} {sign}{delta:>+.4f}")

    # Localization
    for bucket in ["all", "curated", "scrawld"]:
        lA = results_A["localization"].get(bucket, {})
        lB = results_B["localization"].get(bucket, {})
        tA = lA.get("total", 0)
        tB = lB.get("total", 0)
        if tA == 0 and tB == 0:
            continue
        print(f"\n[LOCALIZATION — {bucket.upper()}] (n={max(tA,tB)})")
        print(f"  {'Metric':<15} {'Pretrained':>12} {'Baseline':>12} {'Δ':>10}")
        print(f"  {'-'*49}")
        for m in ["top1", "top3", "top5", "top10"]:
            vA = lA.get(m, 0)
            vB = lB.get(m, 0)
            delta = vA - vB
            sign = "+" if delta >= 0 else ""
            print(f"  {m.upper():<15} {_fmt(vA):>12} {_fmt(vB):>12} {sign}{delta:>+.4f}")

    # Save JSON
    report = {
        "pretrained": results_A,
        "baseline": results_B,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "seed": args.seed,
            "exe_mlm_ckpt": args.exe_mlm_ckpt,
        },
    }
    report_dir = os.path.join(PROJECT_ROOT, "Eval_Phase2/reports")
    os.makedirs(report_dir, exist_ok=True)
    rp = os.path.join(report_dir, "pretrained_vs_baseline_comparison.json")
    with open(rp, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nReport saved: {rp}")


if __name__ == "__main__":
    main()
