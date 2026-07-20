#!/usr/bin/env python3
"""
LLVT Evaluator: Training + Evaluation for Line-Level Vulnerability Tagger.

Key fixes vs graph_mil_evaluator:
1. Direct line-level supervision (not bag-level MIL)
2. Fixed annotation loading: SBCurated ALWAYS included in test split
3. Noise-aware confidence weighting (SBCurated=1.0, ScrawlD=0.3)
4. Focal loss for extreme class imbalance
5. Contrastive loss for better embeddings
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer
import argparse
import sys
from tqdm import tqdm
import numpy as np
import random

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = "./outputs/Eval_Phase2"
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from evaluation.phase2.mil_loader import MILLoader
from models.llvt_model import LLVT
from models.wire.wire_rope_model import WIREModernBERT
from models.wire.wire_rope import WIREConfig


def extract_hidden_states(enc_out):
    """Robustly extract last hidden state from encoder output."""
    # HuggingFace BaseModelOutput
    if hasattr(enc_out, 'last_hidden_state') and enc_out.last_hidden_state is not None:
        return enc_out.last_hidden_state
    # Plain dict (WIRE model returns {"hidden_states": tensor, ...})
    if isinstance(enc_out, dict):
        if "hidden_states" in enc_out and enc_out["hidden_states"] is not None:
            return enc_out["hidden_states"]
        if "last_hidden_state" in enc_out and enc_out["last_hidden_state"] is not None:
            return enc_out["last_hidden_state"]
        # Fallback: first tensor value
        for v in enc_out.values():
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                return v
    # Tuple output
    if isinstance(enc_out, (tuple, list)):
        return enc_out[0]
    raise ValueError(f"Cannot extract hidden states from {type(enc_out)}")


def get_llvt_splits(label_files, annotation_files, seed=42, zero_shot=False, reverse_zero_shot=False):
    """
    FIXED split: ensures SBCurated files are represented in test.
    - All SBCurated files with annotations → 70% train / 15% val / 15% test
    - ScrawlD files → 70% train / 15% val / 15% test
    - If zero_shot=True, Train/Val = ScrawlD only, Test = SBCurated only.
    - If reverse_zero_shot=True, Train/Val = SBCurated only, Test = ScrawlD only.
    """
    rng = random.Random(seed)

    all_data = []
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))

    # Build annotation lookup with dataset source
    curated_filenames = set()
    for af in annotation_files:
        if not os.path.exists(af):
            continue
        with open(af, 'r') as f:
            for line in f:
                d = json.loads(line)
                dataset_tag = d.get("dataset")
                if dataset_tag == "SBCurated" or dataset_tag == "SBCurated_Obfuscated":
                    curated_filenames.add(d["filename"])

    # Separate indices
    curated_indices = [i for i, d in enumerate(all_data) if d["filename"] in curated_filenames]
    other_indices = [i for i, d in enumerate(all_data) if d["filename"] not in curated_filenames]

    rng.shuffle(curated_indices)
    rng.shuffle(other_indices)

    def split_list(lst, train_frac=0.70, val_frac=0.15):
        n = len(lst)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * val_frac))
        return lst[n_val + n_test:], lst[:n_val], lst[n_val:n_val + n_test]

    cur_train, cur_val, cur_test = split_list(curated_indices)
    oth_train, oth_val, oth_test = split_list(other_indices)

    if zero_shot:
        train = oth_train + oth_test
        val = oth_val
        test = curated_indices
    elif reverse_zero_shot:
        train = cur_train + cur_test
        val = cur_val
        test = other_indices
    else:
        train = cur_train + oth_train
        val = cur_val + oth_val
        test = cur_test + oth_test

    rng.shuffle(train)
    
    n_cur_test = len(set(test).intersection(set(curated_indices)))
    n_oth_test = len(set(test).intersection(set(other_indices)))
    print(f"Split: train={len(train)} val={len(val)} test={len(test)} "
          f"(test: {n_cur_test} curated + {n_oth_test} other)")

    return train, val, test


def build_line_labels(filename, annotations, curated_files, num_lines):
    """
    Build per-line binary labels and confidence weights.
    Returns:
        labels: [L] tensor, 1 for vulnerable lines
        weights: [L] tensor, confidence weight per line
        is_curated: bool
    """
    labels = torch.zeros(num_lines)
    weights = torch.ones(num_lines)

    is_curated = filename in curated_files

    if filename in annotations:
        gt_lines = annotations[filename]
        # Confidence: SBCurated = 1.0, ScrawlD = 0.3
        confidence = 1.0 if is_curated else 0.3
        weights[:] = confidence

        for line_idx in gt_lines:
            if 0 < line_idx < num_lines:
                labels[line_idx] = 1.0
                # Vulnerable lines always get full weight
                weights[line_idx] = 1.0

        # Label smoothing for noisy ScrawlD
        if not is_curated:
            labels = labels * 0.85 + 0.05 * (1 - labels) * (labels == 0).float()
    else:
        weights[:] = 0.0  # No annotations → no supervision

    return labels, weights, is_curated


def train_llvt(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if args.label_file:
        label_files = [args.label_file]
    else:
        label_files = [
            os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl"),
            os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_labels.jsonl"),
        ]
        
    if args.annotation_file:
        annotation_files = [args.annotation_file]
    else:
        annotation_files = [
            os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl"),
            os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_annotations.jsonl"),
        ]

    # Load annotations with source info
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
                # SBCurated always takes priority
                if ds == "SBCurated" or fn not in annotations:
                    annotations[fn] = d["vulnerable_lines"]
                if ds == "SBCurated":
                    curated_files.add(fn)

    print(f"Annotations loaded: {len(annotations)} files ({len(curated_files)} curated)")

    # Dataset
    full_dataset = MILLoader(
        label_files, annotation_files, tokenizer,
        max_len=args.max_len, npz_dir=args.npz_dir, df_dir=args.df_dir
    )
    # Override annotations with our priority-resolved version
    full_dataset.annotations = annotations

    train_idx, val_idx, test_idx = get_llvt_splits(label_files, annotation_files, zero_shot=args.zero_shot, reverse_zero_shot=args.reverse_zero_shot)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1)

    # Encoder
    if args.model_type == "wire":
        encoder = WIREModernBERT(args.base_model_path, wire_config=WIREConfig(d_head=64))
    else:
        encoder = AutoModel.from_pretrained(args.base_model_path)
        
    if args.checkpoint:
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter out MLM head or other keys if necessary
        if any(k.startswith("encoder.") for k in sd.keys()):
             sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        elif any(k.startswith("backbone.") for k in sd.keys()):
             sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
             
        encoder.load_state_dict(sd, strict=False)

    hidden_dim = getattr(encoder, "hidden_size", None)
    if hidden_dim is None and hasattr(encoder, "config"):
        hidden_dim = encoder.config.hidden_size
    if hidden_dim is None:
        hidden_dim = 768

    # LLVT head
    llvt = LLVT(
        hidden_dim=hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=4,
        dropout=args.dropout,
        focal_gamma=2.0,
        focal_alpha=0.25,
        contrastive_weight=args.contrastive_weight,
    )

    encoder.to(device)
    llvt.to(device)

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr},
        {"params": llvt.parameters(), "lr": args.lr * 5},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training: {args.epochs} epochs, lr={args.lr}, contrastive_weight={args.contrastive_weight}")

    best_val_metric = 0
    global_step = 0

    for epoch in range(args.epochs):
        encoder.train()
        llvt.train()
        epoch_line_loss = 0
        epoch_bag_loss = 0
        epoch_con_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bag_labels = batch["labels"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]

            # Encode
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                else:
                    # Text-only ablation: provide zeroed graph features
                    kwargs["graph_features"] = torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
                enc_out = encoder(**kwargs)
            else:
                enc_out = encoder(input_ids, attention_mask=attention_mask)

            tok_emb = extract_hidden_states(enc_out)

            B = input_ids.size(0)
            max_line = token_to_line.max().item()
            L = max(max_line + 1, 2)

            # Build line-level labels
            all_line_labels = torch.zeros(B, L, device=device)
            all_line_weights = torch.ones(B, L, device=device)

            for b in range(B):
                ll, ww, _ = build_line_labels(
                    filenames[b], annotations, curated_files, L
                )
                all_line_labels[b] = ll.to(device)
                all_line_weights[b] = ww.to(device)

            # LLVT forward
            out = llvt(
                tok_emb, token_to_line, attention_mask,
                line_labels=all_line_labels,
                line_label_weights=all_line_weights,
                bag_labels=bag_labels,
            )

            loss = out["line_loss"] + 0.3 * out.get("bag_loss", 0) + args.contrastive_weight * out.get("contrastive_loss", 0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(llvt.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_line_loss += out["line_loss"].item()
            epoch_bag_loss += out.get("bag_loss", torch.tensor(0)).item()
            epoch_con_loss += out.get("contrastive_loss", torch.tensor(0)).item()
            n_batches += 1
            global_step += 1

        avg_l = epoch_line_loss / n_batches
        avg_b = epoch_bag_loss / n_batches
        avg_c = epoch_con_loss / n_batches
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | line={avg_l:.4f} bag={avg_b:.4f} con={avg_c:.4f} | lr={lr_now:.2e}")

        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            val_metric = validate_llvt(encoder, llvt, val_loader, device, args, annotations, curated_files)
            print(f"  Val Top-5 Recall: {val_metric:.4f}")

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                ckpt_dir = os.path.join(OUTPUT_ROOT, f"checkpoints/{args.model_name}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "encoder": encoder.state_dict(),
                    "llvt": llvt.state_dict(),
                    "epoch": epoch,
                    "val_metric": val_metric,
                }, os.path.join(ckpt_dir, "best.pt"))
                print(f"  Saved best checkpoint (val_top5={val_metric:.4f})")

    # Final evaluation
    evaluate_llvt(encoder, llvt, test_loader, device, args, annotations, curated_files)


def validate_llvt(encoder, llvt, val_loader, device, args, annotations, curated_files):
    encoder.eval()
    llvt.eval()
    hits_5 = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filenames = batch["filename"]

            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                else:
                    kwargs["graph_features"] = torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
                enc_out = encoder(**kwargs)
            else:
                enc_out = encoder(input_ids, attention_mask=attention_mask)

            tok_emb = extract_hidden_states(enc_out)

            out = llvt(tok_emb, token_to_line, attention_mask)

            for b in range(input_ids.size(0)):
                fn = filenames[b]
                if fn not in annotations:
                    continue
                gt = annotations[fn]
                scores = out["line_scores"][b].cpu().numpy()
                mask = out["line_mask"][b].cpu().numpy()

                line_scores = []
                for li in range(len(scores)):
                    if mask[li] > 0 and li > 0:
                        line_scores.append((li, scores[li]))
                top5 = [l for l, s in sorted(line_scores, key=lambda x: x[1], reverse=True)[:5]]

                total += 1
                if any(l in gt for l in top5):
                    hits_5 += 1

    return hits_5 / max(1, total)


def evaluate_llvt(encoder, llvt, test_loader, device, args, annotations, curated_files):
    encoder.eval()
    llvt.eval()

    results = {
        "all": {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
        "curated": {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
        "scrawld": {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "total": 0},
    }

    vis_dir = os.path.join(OUTPUT_ROOT, f"localization_results/{args.model_name}")
    os.makedirs(vis_dir, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            fn = batch["filename"][0]

            if fn not in annotations:
                continue
            gt = annotations[fn]
            is_curated = fn in curated_files

            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                else:
                    kwargs["graph_features"] = torch.zeros((input_ids.size(0), input_ids.size(1), 6), device=device)
                enc_out = encoder(**kwargs)
            else:
                enc_out = encoder(input_ids, attention_mask=attention_mask)

            tok_emb = extract_hidden_states(enc_out)

            out = llvt(tok_emb, token_to_line, attention_mask)

            scores = out["line_scores"].squeeze(0).cpu().numpy()
            mask = out["line_mask"].squeeze(0).cpu().numpy()

            line_scores = [(li, float(scores[li])) for li in range(len(scores)) if mask[li] > 0 and li > 0]
            sorted_lines = sorted(line_scores, key=lambda x: x[1], reverse=True)
            top_lines = [l for l, s in sorted_lines]

            # Update results
            for bucket in ["all", "curated" if is_curated else "scrawld"]:
                r = results[bucket]
                r["total"] += 1
                if top_lines and top_lines[0] in gt:
                    r["top1"] += 1
                if any(l in gt for l in top_lines[:3]):
                    r["top3"] += 1
                if any(l in gt for l in top_lines[:5]):
                    r["top5"] += 1
                if any(l in gt for l in top_lines[:10]):
                    r["top10"] += 1

            # Save visualizations
            if vis_count < 50:
                safe_fn = fn.replace("/", "_").replace("\\", "_")
                with open(os.path.join(vis_dir, f"{safe_fn}_vis.txt"), 'w') as f:
                    f.write(f"Dataset: {'SBCurated' if is_curated else 'ScrawlD'}\n")
                    f.write(f"GT Lines: {gt}\n")
                    f.write(f"Top 20 Predictions:\n")
                    for rank, (li, sc) in enumerate(sorted_lines[:20]):
                        hit = " [HIT]" if li in gt else ""
                        f.write(f"  {rank+1}. Line {li}: {sc:.6f}{hit}\n")
                vis_count += 1

    # Print
    print(f"\n{'='*60}")
    print(f"  LLVT Localization Results: {args.model_name}")
    print(f"{'='*60}")

    for name, r in results.items():
        if r["total"] > 0:
            print(f"\n  [{name.upper()}] Total: {r['total']}")
            print(f"    Top-1:  {r['top1']/r['total']:.4f} ({r['top1']}/{r['total']})")
            print(f"    Top-3:  {r['top3']/r['total']:.4f}")
            print(f"    Top-5:  {r['top5']/r['total']:.4f}")
            print(f"    Top-10: {r['top10']/r['total']:.4f}")

    # Save report
    report = {"model_name": args.model_name}
    for name, r in results.items():
        t = max(1, r["total"])
        report[name] = {k: v/t if k != "total" else v for k, v in r.items()}

    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    rp = os.path.join(report_dir, f"{args.model_name}_localization.json")
    with open(rp, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nReport: {rp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", required=True, choices=["wire", "modernbert", "codebert", "graphcodebert"])
    parser.add_argument("--base_model_path", default="microsoft/codebert-base")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--npz_dir", default=None)
    parser.add_argument("--df_dir", default=None)
    parser.add_argument("--label_file", default=None, help="Path to labels.jsonl")
    parser.add_argument("--annotation_file", default=None, help="Path to annotations.jsonl")
    parser.add_argument("--num_gat_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--zero_shot", action="store_true", help="If set, train on ScrawlD and test exclusively on SBCurated")
    parser.add_argument("--reverse_zero_shot", action="store_true", help="If set, train on SBCurated and test exclusively on ScrawlD")
    args = parser.parse_args()
    train_llvt(args)
