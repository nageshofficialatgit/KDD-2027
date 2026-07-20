#!/usr/bin/env python3
"""
Comprehensive Multi-Model Comparison: Pre-trained vs. Untrained Baselines.
Evaluates e1_err, wire_abl, and exe_mlm against their respective untrained variants on LLVT.
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
from models.baseline.graph_rope_model import GraphRoPEModernBERT
from models.baseline.graph_rope import GraphRoPEConfig


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
    if isinstance(enc_out, (tuple, list)):
        return enc_out[0]
    raise ValueError(f"Cannot extract hidden states from {type(enc_out)}")


def get_splits(label_files, annotation_files, seed=42):
    rng = random.Random(seed)
    all_data = []
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))

    curated_filenames = set()
    for af in annotation_files:
        if not os.path.exists(af): continue
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
    return ct + ot, cv + ov, ce + oe


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
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    encoder, llvt, train_loader, val_loader, test_loader,
    annotations, curated_files, device, tag, epochs=10, lr=2e-5
):
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr},
        {"params": llvt.parameters(), "lr": lr * 5},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = 0
    best_state = None

    for epoch in range(epochs):
        encoder.train(); llvt.train()
        ep_loss = 0; n = 0

        for batch in tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bag_labels = batch["labels"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]

            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            kwargs["graph_features"] = g_feats.to(device) if g_feats is not None else torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
            
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

            out = llvt(tok_emb, token_to_line, attention_mask, line_labels=all_line_labels, line_label_weights=all_line_weights, bag_labels=bag_labels)
            loss = out["line_loss"] + 0.3 * out.get("bag_loss", 0) + 0.1 * out.get("contrastive_loss", 0)
            
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(llvt.parameters()), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item(); n += 1

        avg = ep_loss / max(n, 1)
        if (epoch + 1) % 2 == 0:
            val_met = _validate(encoder, llvt, val_loader, device, annotations, curated_files)
            print(f"[{tag}] Epoch {epoch+1}/{epochs} loss={avg:.4f} val_top5={val_met:.4f}")
            if val_met > best_val:
                best_val = val_met
                best_state = {"encoder": copy.deepcopy(encoder.state_dict()), "llvt": copy.deepcopy(llvt.state_dict())}
        else:
            print(f"[{tag}] Epoch {epoch+1}/{epochs} loss={avg:.4f}")

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        llvt.load_state_dict(best_state["llvt"])

    return _evaluate(encoder, llvt, test_loader, device, annotations, curated_files)


def _validate(encoder, llvt, loader, device, annotations, curated_files):
    encoder.eval(); llvt.eval()
    hits = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]
            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            kwargs["graph_features"] = g_feats.to(device) if g_feats is not None else torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
            
            enc_out = encoder(**kwargs)
            tok_emb = extract_hidden_states(enc_out)
            out = llvt(tok_emb, token_to_line, attention_mask)

            for b in range(input_ids.size(0)):
                fn = filenames[b]
                if fn not in annotations: continue
                gt = annotations[fn]
                scores = out["line_scores"][b].cpu().numpy()
                mask = out["line_mask"][b].cpu().numpy()
                ls = [(li, scores[li]) for li in range(len(scores)) if mask[li] > 0 and li > 0]
                top5 = [l for l, s in sorted(ls, key=lambda x: x[1], reverse=True)[:5]]
                total += 1
                if any(l in gt for l in top5): hits += 1
    return hits / max(1, total)


def _evaluate(encoder, llvt, loader, device, annotations, curated_files):
    encoder.eval(); llvt.eval()
    results = {
        "all":     {"top1": 0, "top3": 0, "top5": 0, "total": 0},
        "curated": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
        "scrawld": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
    }
    detection = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            bag_labels = batch["labels"].to(device)
            fn = batch["filename"][0]

            g_feats = batch.get("graph_features")
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            kwargs["graph_features"] = g_feats.to(device) if g_feats is not None else torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
            
            enc_out = encoder(**kwargs)
            tok_emb = extract_hidden_states(enc_out)
            out = llvt(tok_emb, token_to_line, attention_mask)

            bag_probs = torch.sigmoid(out["bag_logits"]).squeeze(0)
            is_vuln = bag_labels.max() > 0
            pred_vuln = bag_probs.max() > 0.5
            if is_vuln and pred_vuln:       detection["tp"] += 1
            elif not is_vuln and pred_vuln: detection["fp"] += 1
            elif is_vuln and not pred_vuln: detection["fn"] += 1
            else:                           detection["tn"] += 1

            if fn not in annotations: continue
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

    p = detection["tp"] / (detection["tp"] + detection["fp"] + 1e-8)
    r = detection["tp"] / (detection["tp"] + detection["fn"] + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return {
        "localization": {n: {k: v / max(1, vals["total"]) if k != "total" else v for k, v in vals.items()} for n, vals in results.items()},
        "detection": {**detection, "f1": f1},
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip_exe", action="store_true")
    parser.add_argument("--model", type=str, choices=["6D", "8D", "10D", "all"], default="all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    base_model = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

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
        if not os.path.exists(af): continue
        with open(af, 'r') as f:
            for line in f:
                d = json.loads(line)
                fn = d["filename"]
                if d.get("dataset") == "SBCurated" or fn not in annotations:
                    annotations[fn] = d["vulnerable_lines"]
                if d.get("dataset") == "SBCurated": curated_files.add(fn)

    models_to_run = [
        {
            "name": "6D_Baseline",
            "type": "wire",
            "ckpt": os.path.join(PROJECT_ROOT, "checkpoints/wire_abl/best.pt"),
            "n_features": 6,
            "npz_dir": "Implementation/evaluation/cache/graphs/eval_shards"
        },
        {
            "name": "8D_Ablation",
            "type": "wire",
            "ckpt": os.path.join(PROJECT_ROOT, "checkpoints/wire_abl_8d/best.pt"),
            "n_features": 8,
            "npz_dir": "Implementation/evaluation/cache/graphs/eval_shards_8d"
        },
        {
            "name": "10D_Ablation",
            "type": "wire",
            "ckpt": os.path.join(PROJECT_ROOT, "checkpoints/wire_abl_10d/best.pt"),
            "n_features": 10,
            "npz_dir": "Implementation/evaluation/cache/graphs/eval_shards_10d"
        },
    ]

    if args.model != "all":
        # Match using beginning of name (e.g. "6D")
        models_to_run = [m for m in models_to_run if m["name"].startswith(args.model)]

    full_results = {}

    for mcfg in models_to_run:
        name = mcfg["name"]
        print(f"\n\n>>> RUNNING COMPARISON FOR: {name} <<<")
        
        dataset = MILLoader(label_files, annotation_files, tokenizer, max_len=512, 
                             npz_dir=os.path.join(PROJECT_ROOT, mcfg["npz_dir"]))
        dataset.annotations = annotations
        train_idx, val_idx, test_idx = get_splits(label_files, annotation_files)
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1)
        
        # 1. PRE-TRAINED
        seed_everything(42)
        if mcfg["type"] == "wire":
            encoder = WIREModernBERT(base_model, wire_config=WIREConfig(d_head=64, n_features=mcfg["n_features"]))
        else:
            encoder = GraphRoPEModernBERT(base_model, rope_config=GraphRoPEConfig(d_head=64))
        
        print(f"Loading checkpoint: {mcfg['ckpt']}")
        ckpt = torch.load(mcfg["ckpt"], map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        
        # Generic weight loading
        encoder_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder."): encoder_sd[k.replace("encoder.", "")] = v
        
        target_encoder = encoder.encoder if hasattr(encoder, "encoder") else encoder
        missing, unexpected = target_encoder.load_state_dict(encoder_sd, strict=False)
        print(f"Loaded encoder weights: {len(encoder_sd)} keys. Missing={len(missing)}, Unexpected={len(unexpected)}")

        # Load RoPE weights
        rope_key = "wire_rope." if mcfg["type"] == "wire" else "graph_rope."
        rope_sd = {k.replace(rope_key, ""): v for k, v in sd.items() if k.startswith(rope_key)}
        if rope_sd:
            target_rope = encoder.wire_rope if mcfg["type"] == "wire" else encoder.graph_rope
            target_rope.load_state_dict(rope_sd, strict=False)
            print(f"Loaded {rope_key} weights: {len(rope_sd)} keys")

        llvt = LLVT(hidden_dim=768).to(device)
        encoder.to(device)
        
        res_pt = train_and_evaluate(encoder, llvt, train_loader, val_loader, test_loader, annotations, curated_files, device, f"{name}_PT", epochs=args.epochs, lr=args.lr)
        
        # Save PT results immediately
        pt_path = os.path.join(PROJECT_ROOT, f"Eval_Phase2/reports/partial_{name}_pt.json")
        with open(pt_path, 'w') as f:
            json.dump(res_pt, f, indent=4)

        # 2. BASELINE (Untrained)
        print(f"Running Untrained Baseline for {name}...")
        seed_everything(42)
        if mcfg["type"] == "wire":
            encoder_base = WIREModernBERT(base_model, wire_config=WIREConfig(d_head=64, n_features=mcfg["n_features"]))
        else:
            encoder_base = GraphRoPEModernBERT(base_model, rope_config=GraphRoPEConfig(d_head=64))
        
        llvt_base = LLVT(hidden_dim=768).to(device)
        encoder_base.to(device)
        
        res_base = train_and_evaluate(encoder_base, llvt_base, train_loader, val_loader, test_loader, annotations, curated_files, device, f"{name}_BASE", epochs=args.epochs, lr=args.lr)

        full_results[name] = {
            "pretrained": res_pt,
            "baseline": res_base
        }

        # Save partial result immediately if running in parallel
        if args.model != "all":
            part_path = os.path.join(PROJECT_ROOT, f"Eval_Phase2/reports/comparison_{name}.json")
            with open(part_path, 'w') as f:
                json.dump({name: full_results[name]}, f, indent=4)
            print(f"Partial report saved: {part_path}")

    # Save final report if all are run
    if args.model == "all":
        out_path = os.path.join(PROJECT_ROOT, "Eval_Phase2/reports/multi_model_comparison.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(full_results, f, indent=4)
        print(f"\nFinal report saved to: {out_path}")

if __name__ == "__main__":
    main()
