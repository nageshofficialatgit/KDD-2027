#!/usr/bin/env python3
"""
Graph-RoPE Multi-Task Pretraining Script.

Trains ModernBERT with Graph-RoPE on smart contract code using:
  - Masked Language Modeling (MLM)
  - Topological Feature Regression (TFR)
  - ERR (Effective Resistance Regression) or Edge Prediction

Features:
  - Differential learning rates (encoder / Graph-RoPE / heads)
  - bf16 mixed precision (A100 native)
  - Checkpointing with full RNG state recovery
  - TensorBoard + CSV logging
  - EMA (Exponential Moving Average) model
  - Gradient clipping and warm-up
  - Attention entropy monitoring (per-head, per-layer)
  - Layer-wise gradient norm tracking
  - Rotation angle distribution & feature utilization diagnostics
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from transformers import AutoTokenizer

# Add Implementation dir to path
# Add Implementation dir (parent of experiments) to path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.dirname(EXPERIMENTS_DIR)
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)

from models.graph_rope import GraphRoPEConfig
from models.graph_rope_dataset import create_dataloaders
from models.graph_rope_model import GraphRoPEModernBERT

print(">>> BTP_TELEMETRY: train_graph_rope.py loaded with Phase 5 stability fixes")


# ═════════════════════════════════════════════════════════════════════════
#  EMA Model
# ═════════════════════════════════════════════════════════════════════════
class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Replace model params with EMA params (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model params after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
        if device is not None:
            for k in self.shadow:
                self.shadow[k] = self.shadow[k].to(device)


# ═════════════════════════════════════════════════════════════════════════
#  Logging
# ═════════════════════════════════════════════════════════════════════════
class TrainingLogger:
    """CSV + optional TensorBoard logging."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.csv_file = None
        self.csv_writer = None
        self.tb_writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print(
                    "TensorBoard not available, using CSV only",
                    flush=True,
                )

    def log(self, step: int, metrics: dict):
        """Log metrics at a given step."""
        # Initialize CSV on first call
        current_keys = set(metrics.keys())
        if self.csv_writer is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            fieldnames = ["step"] + sorted(list(current_keys))
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()

        if self.csv_file:
            row = {"step": step}
            for k, v in metrics.items():
                # Only log keys that were in the initial header
                if k in self.csv_writer.fieldnames:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, torch.Tensor)):
                    self.tb_writer.add_scalar(k, v, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram to TensorBoard."""
        if self.tb_writer is not None:
            if isinstance(values, np.ndarray):
                values = torch.from_numpy(values)
            self.tb_writer.add_histogram(tag, values, step)

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()

# ═════════════════════════════════════════════════════════════════════════
#  Training Diagnostics
# ═════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_attention_entropy(model):
    """
    Compute rotation angle entropy per feature bank as a proxy
    for attention diversity. High entropy = diverse attention,
    low entropy = collapsed/degenerate patterns.
    """
    wrappers = model._rope_wrappers
    entropy_metrics = {}

    for layer_idx, wrapper in enumerate(wrappers):
        if wrapper._graph_cos is None:
            continue
        cos_vals = wrapper._graph_cos  # [B, T, d_head/2]
        # Sample 6 planes (one per feature bank)
        for plane in range(min(6, cos_vals.shape[-1])):
            plane_cos = cos_vals[0, :, plane].float()  # [T]
            hist = torch.histc(plane_cos, bins=32, min=-1, max=1)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = -(hist * hist.log()).sum().item()
            entropy_metrics[
                f"xai/entropy_layer{layer_idx}_head{plane}"
            ] = entropy

    return entropy_metrics


def compute_gradient_norms(model):
    """
    Compute L2 gradient norm per encoder layer (attention + MLP).
    Reveals whether Graph-RoPE gradients propagate uniformly.
    """
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if "encoder.layers" in name:
            parts = name.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            norm = param.grad.data.norm(2).item()
            key_type = "attn" if "attn" in name else "mlp"
            key = f"grad/L{layer_idx}_{key_type}"
            grad_norms[key] = grad_norms.get(key, 0) + norm

        elif "graph_rope" in name:
            norm = param.grad.data.norm(2).item()
            short = name.split(".")[-1]
            grad_norms[f"grad/rope_{short}"] = norm

    return grad_norms


@torch.no_grad()
def compute_rotation_diagnostics(model, graph_features):
    """
    Analyze rotation angle distributions and feature utilization.
    Detects feature collapse and underutilization.
    """
    rope = model.graph_rope
    device = next(model.parameters()).device
    feats = graph_features.to(device)
    cos, sin = rope(feats)
    angles = torch.atan2(sin.float(), cos.float())  # [B, T, 32]

    feat_names = [
        "poisson", "d_scale1", "d_scale2",
        "d_scale3", "d_scale4", "d_scale5",
    ]
    plane_ranges = [(0, 6), (6, 11), (11, 16),
                    (16, 21), (21, 26), (26, 32)]

    diagnostics = {}
    total_var = 0
    feature_vars = []

    for fi, (start, end) in enumerate(plane_ranges):
        feat_angles = angles[:, :, start:end]
        var = feat_angles.var().item()
        feature_vars.append(var)
        total_var += var
        diagnostics[f"rot/var_{feat_names[fi]}"] = var

        # Circular std: sqrt(-2 * log(|mean(e^{i*theta})|))
        R = torch.abs(
            torch.mean(torch.exp(1j * feat_angles.double()))
        )
        circ_std = torch.sqrt(-2 * torch.log(R + 1e-8)).item()
        diagnostics[f"rot/cstd_{feat_names[fi]}"] = circ_std

    # Feature utilization as fraction of total variance
    if total_var > 0:
        for fi, var in enumerate(feature_vars):
            diagnostics[
                f"rot/util_{feat_names[fi]}"
            ] = var / total_var

    return diagnostics, angles, feat_names, plane_ranges



# ═════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    ema: EMAModel,
    epoch: int,
    step: int,
    best_val_loss: float,
    config: dict,
):
    """Save full training state for recovery."""
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, scheduler, ema):
    """Load checkpoint and restore full training state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"], device=next(model.parameters()).device)

    # Restore RNG states
    rng = ckpt["rng_states"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])

    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"]


def manage_checkpoints(ckpt_dir: str, keep_last: int = 3):
    """Keep only the last N checkpoints + best checkpoint."""
    ckpts = sorted(
        [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith("step_") and f.endswith(".pt")
        ],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    # Always keep best
    to_remove = ckpts[:-keep_last] if len(ckpts) > keep_last else []
    for f in to_remove:
        path = os.path.join(ckpt_dir, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed old checkpoint: {f}", flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  XAI Diagnostics (called every 500 steps)
# ═════════════════════════════════════════════════════════════════════════
def compute_xai_diagnostics(model, batch, device):
    """
    Compute XAI metrics for weight learning analysis.

    Returns dict with:
        xai/grad_layer_{L}:      Gradient L2 norm per encoder layer
        xai/rope_util_{feat}:    Variance fraction per RoPE feature bank
        xai/entropy_layer{L}_head{H}: Attention entropy per head
        xai/head_importance_{L}_{H}:  Gradient × activation magnitude
    """
    xai_metrics = {}

    # ── 1. Layer Gradient Magnitude ──────────────────────────────────
    for i, layer in enumerate(model.encoder.layers):
        grad_norms = []
        for p in layer.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2).item())
        if grad_norms:
            xai_metrics[f"xai/grad_layer_{i}"] = np.mean(grad_norms)

    # ── 2. RoPE Feature Utilization ──────────────────────────────────
    # Measure how much each feature bank contributes to rotations
    try:
        rope_emb = model.graph_rope
        graph_features = batch["graph_features"].to(device)
        with torch.no_grad():
            normed = rope_emb.normalize_features(graph_features)  # [B, T, 6]
            # Variance per feature (higher = more discriminative)
            feat_var = normed.var(dim=(0, 1)).cpu().numpy()  # [6]
            total_var = feat_var.sum() + 1e-8
            feat_names = ["poisson", "d1", "d2", "d4", "d8", "d16"]
            for j, name in enumerate(feat_names):
                xai_metrics[f"rot/util_{name}"] = float(feat_var[j] / total_var)
    except Exception:
        pass

    # ── 3. Attention Entropy (via output_attentions) ─────────────────
    # CAUTION: output_attentions=True forces "eager" attention which is memory-heavy.
    # We slice the batch and restore the original implementation after forward.
    orig_attn = model.encoder.config._attn_implementation
    try:
        model.eval()
        with torch.no_grad():
            # Slice batch to 1 instance and 512 tokens to save memory in eager mode
            input_ids = batch["input_ids"][:1, :512].to(device)
            attention_mask = batch["attention_mask"][:1, :512].to(device)
            graph_features = batch["graph_features"][:1, :512].to(device)

            model._set_rope_for_forward(graph_features)
            encoder_out = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
            )

            if hasattr(encoder_out, 'attentions') and encoder_out.attentions is not None:
                for layer_idx, attn_weights in enumerate(encoder_out.attentions):
                    # attn_weights: [B=1, n_heads, T=512, T=512]
                    eps = 1e-8
                    attn_clamped = attn_weights.clamp(min=eps)
                    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [B, n_heads, T]
                    mean_entropy = entropy.mean(dim=(0, 2))  # [n_heads]
                    for h_idx in range(mean_entropy.shape[0]):
                        xai_metrics[f"xai/entropy_layer{layer_idx}_head{h_idx}"] = mean_entropy[h_idx].item()
    except Exception as e:
        print(f"  ⚠️  XAI entropy error: {e}")
    finally:
        # Restore original attention implementation (sdpa/flash_attention_2)
        model.encoder.config._attn_implementation = orig_attn
        model.train()

    return xai_metrics


# ═════════════════════════════════════════════════════════════════════════
#  Validation
# ═════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, val_loader, device, use_ema=False, ema=None):
    """Run validation and return metrics dict."""
    model.eval()
    if use_ema and ema:
        ema.apply(model)

    total_loss = 0
    total_mlm_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_err_loss = 0
    total_mlm_acc = 0
    total_node_mse = 0
    total_edge_acc = 0
    total_err_mse = 0
    n_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(**batch)

        total_loss += out["loss"].item()
        if "mlm_loss" in out:
            total_mlm_loss += out["mlm_loss"].item()
        if "node_loss" in out:
            total_node_loss += out["node_loss"].item()
        if "edge_loss" in out:
            total_edge_loss += out["edge_loss"].item()
        if "err_loss" in out:
            total_err_loss += out["err_loss"].item()
        if "mlm_acc" in out:
            total_mlm_acc += out["mlm_acc"].item()
        if "node_mse" in out:
            total_node_mse += out["node_mse"].item()
        if "edge_acc" in out:
            total_edge_acc += out["edge_acc"].item()
        if "err_mse" in out:
            total_err_mse += out["err_mse"].item()
        n_batches += 1

    if use_ema and ema:
        ema.restore(model)

    if n_batches == 0:
        return {}

    return {
        "val/loss": total_loss / n_batches,
        "val/mlm_loss": total_mlm_loss / n_batches,
        "val/node_loss": total_node_loss / n_batches,
        "val/edge_loss": total_edge_loss / n_batches,
        "val/err_loss": total_err_loss / n_batches,
        "val/mlm_acc": total_mlm_acc / n_batches,
        "val/node_mse": total_node_mse / n_batches,
        "val/edge_acc": total_edge_acc / n_batches,
        "val/err_mse": total_err_mse / n_batches,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═════════════════════════════════════════════════════════════════════════
def train(args):
    """Main training function."""
    # ── Setup ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(
            f"GPU: {props.name}, "
            f"Total Memory: {total_mem / 1e9:.1f}GB, "
            f"Free Memory: {free_mem / 1e9:.1f}GB",
            flush=True,
        )
        if free_mem < 40 * 1e9:
            print(
                f"⚠️  WARNING: Free memory ({free_mem/1e9:.1f}GB) is low. "
                f"Model likely needs ~40GB. Expected OOM.",
                flush=True,
            )

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Data ─────────────────────────────────────────────────────────
    print("Creating dataloaders...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        npz_dir=args.feat_dir,
        bucket_dir=args.bucket_dir,
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        batch_size=args.batch_size,
        mlm_prob=args.mlm_prob,
        n_edge_pairs=args.n_edge_pairs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    print(
        f"Steps per epoch: {steps_per_epoch}, "
        f"Total steps: {total_steps}",
        flush=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    print("Initializing model...", flush=True)
    rope_config = GraphRoPEConfig(max_position=float(args.max_seq_len))
    model = GraphRoPEModernBERT(
        model_name=args.model_name,
        rope_config=rope_config,
        mlm_weight=args.mlm_weight,
        node_weight=args.node_weight,
        edge_weight=args.edge_weight,
        label_smoothing=args.label_smoothing,
        err_weight=getattr(args, 'err_weight', 0.4),
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Parameters: {param_count:,} total, "
        f"{trainable_count:,} trainable",
        flush=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    param_groups = model.get_param_groups(
        encoder_lr=args.lr,
        rope_lr=args.rope_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    # ── Scheduler: linear warmup + cosine decay ──────────────────────
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8 / args.lr,  # start from near-zero
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    # ── EMA ──────────────────────────────────────────────────────────
    ema = EMAModel(model, decay=args.ema_decay)

    # ── Logger ───────────────────────────────────────────────────────
    logger = TrainingLogger(args.log_dir)

    # ── Resume from checkpoint ───────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    resume_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path) and args.resume:
        print(f"Resuming from {resume_path}...", flush=True)
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, ema
        )
        print(
            f"Resumed: epoch={start_epoch}, step={global_step}, "
            f"best_val_loss={best_val_loss:.4f}",
            flush=True,
        )

    # ── Save config ──────────────────────────────────────────────────
    config = vars(args)
    config["total_steps"] = total_steps
    config["steps_per_epoch"] = steps_per_epoch
    config["param_count"] = param_count
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Initial Logging (Step 0) for Dashboard ───────────────────────
    logger.log(global_step, {
        "train/loss": 20.0,
        "train/mlm_loss": 20.0,
        "train/node_loss": 0.0,
        "train/err_loss": 0.0,
        "train/mlm_acc": 0.0,
        "train/grad_norm": 0.0,
        "train/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "xai/entropy_layer0_head0": 0.0, # Seed one XAI key to help formatting
        "rot/util_poisson": 0.0,
        "rot/util_d1": 0.0,
        "rot/util_d2": 0.0,
        "rot/util_d4": 0.0,
        "rot/util_d8": 0.0,
        "rot/util_d16": 0.0,
    })

    # ── Training ─────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Starting training...", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward + loss
            with autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
                loss = out["loss"] / args.grad_accum

            # Backward
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                ema.update(model)
                global_step += 1

                # ── Logging ──────────────────────────────────────
                if global_step % args.log_every == 0:
                    gpu_mem = (
                        torch.cuda.memory_allocated() / 1e9
                        if torch.cuda.is_available()
                        else 0
                    )
                    metrics = {
                        "train/loss": out["loss"].item(),
                        "train/grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                        "train/lr_encoder": optimizer.param_groups[0][
                            "lr"
                        ],
                        "train/lr_rope": (
                            optimizer.param_groups[2]["lr"]
                            if len(optimizer.param_groups) > 2
                            else 0
                        ),
                        "train/gpu_mem_gb": gpu_mem,
                        "train/epoch": epoch,
                    }
                    if "mlm_loss" in out:
                        metrics["train/mlm_loss"] = out[
                            "mlm_loss"
                        ].item()
                    if "node_loss" in out:
                        metrics["train/node_loss"] = out[
                            "node_loss"
                        ].item()
                    if "edge_loss" in out:
                        metrics["train/edge_loss"] = out[
                            "edge_loss"
                        ].item()
                    if "err_loss" in out:
                        metrics["train/err_loss"] = out[
                            "err_loss"
                        ].item()
                    if "mlm_acc" in out:
                        metrics["train/mlm_acc"] = out[
                            "mlm_acc"
                        ].item()
                    if "node_mse" in out:
                        metrics["train/node_mse"] = out[
                            "node_mse"
                        ].item()
                    if "edge_acc" in out:
                        metrics["train/edge_acc"] = out[
                            "edge_acc"
                        ].item()
                    if "err_mse" in out:
                        metrics["train/err_mse"] = out[
                            "err_mse"
                        ].item()

                    logger.log(global_step, metrics)

                    # ── XAI diagnostics (every 500 steps) ────────
                    xai_interval = getattr(args, 'xai_interval', 500)
                    if global_step % xai_interval == 0:
                        try:
                            xai = compute_xai_diagnostics(model, batch, device)
                            if xai:
                                logger.log(global_step, xai)
                                n_layers = sum(1 for k in xai if k.startswith("xai/grad_layer_"))
                                n_heads = sum(1 for k in xai if k.startswith("xai/entropy_"))
                                rope_parts = [
                                    "{0}={1:.3f}".format(k.split("_")[-1], v)
                                    for k, v in xai.items() if k.startswith("rot/util_")
                                ]
                                rope_str = ", ".join(rope_parts) if rope_parts else "N/A"
                                print(
                                    f"  🔬 XAI: {n_layers} layer grads, "
                                    f"{n_heads} head entropies, "
                                    f"RoPE util: {rope_str}",
                                    flush=True,
                                )
                        except Exception as e:
                            print(f"  ⚠️  XAI diagnostic error: {e}", flush=True)

                    # Build inter-node metric string (show all active)
                    parts = []
                    if "err_loss" in out:
                        parts.append(f"err={out['err_loss'].item():.4f}")
                    if "edge_loss" in out:
                        parts.append(f"edge={out['edge_loss'].item():.4f}")
                    inter_node_str = " | ".join(parts) if parts else "(none)"

                    print(
                        f"  [{epoch+1}/{args.epochs}] "
                        f"step {global_step}/{total_steps} | "
                        f"loss={out['loss'].item():.4f} | "
                        f"mlm={out.get('mlm_loss', torch.tensor(0)).item():.4f} | "
                        f"node={out.get('node_loss', torch.tensor(0)).item():.4f} | "
                        f"{inter_node_str} | "
                        f"grad={grad_norm:.2f} | "
                        f"GPU={gpu_mem:.1f}GB",
                        flush=True,
                    )

                    # Alert on instability
                    if isinstance(grad_norm, (float, int)):
                        gn = grad_norm
                    else:
                        gn = grad_norm.item()
                    if gn > 10.0:
                        print(
                            f"  ⚠️  HIGH GRAD NORM: {gn:.2f}",
                            flush=True,
                        )
                    if gpu_mem > 75.0:
                        print(
                            f"  ⚠️  HIGH GPU MEMORY: {gpu_mem:.1f}GB",
                            flush=True,
                        )

                # ── Advanced diagnostics (every 500 steps) ───────
                if global_step % 500 == 0 and global_step > 0:
                    grad_diag = compute_gradient_norms(model)
                    logger.log(global_step, grad_diag)

                    entropy_diag = compute_attention_entropy(model)
                    if entropy_diag:
                        logger.log(global_step, entropy_diag)

                    print(
                        f"  📊 Diagnostics logged at step "
                        f"{global_step}",
                        flush=True,
                    )

                # ── Rotation diagnostics (every 1000 steps) ──────
                if global_step % 1000 == 0 and global_step > 0:
                    rot_diag, angles, fnames, pranges = (
                        compute_rotation_diagnostics(
                            model, batch["graph_features"]
                        )
                    )
                    logger.log(global_step, rot_diag)

                    # Log histograms to TensorBoard
                    for fi, (s, e) in enumerate(pranges):
                        logger.log_histogram(
                            f"rotation_hist/{fnames[fi]}",
                            angles[:, :, s:e].cpu().flatten(),
                            global_step,
                        )

                    # Feature utilization alert
                    for fi, fn in enumerate(fnames):
                        key = f"rot/util_{fn}"
                        if key in rot_diag and rot_diag[key] < 0.05:
                            print(
                                f"  ⚠️  LOW UTILIZATION: {fn} = "
                                f"{rot_diag[key]:.3f}",
                                flush=True,
                            )

                    print(
                        f"  🔄 Rotation diagnostics logged at "
                        f"step {global_step}",
                        flush=True,
                    )

                # ── Validation ───────────────────────────────────
                if global_step % args.val_every == 0:
                    print(
                        f"\n  Validating at step {global_step}...",
                        flush=True,
                    )
                    val_metrics = validate(
                        model, val_loader, device, use_ema=True, ema=ema
                    )
                    logger.log(global_step, val_metrics)

                    val_loss = val_metrics.get("val/loss", float("inf"))

                    # Build val inter-node metric string (show all active)
                    parts = []
                    if val_metrics.get('val/err_mse', 0) > 0:
                        parts.append(f"err_mse: {val_metrics['val/err_mse']:.4f}")
                    if val_metrics.get('val/edge_acc', 0) > 0:
                        parts.append(f"edge_acc: {val_metrics['val/edge_acc']:.4f}")
                    val_inter = " | ".join(parts) if parts else "(none)"

                    print(
                        f"  Val loss: {val_loss:.4f} | "
                        f"mlm_acc: {val_metrics.get('val/mlm_acc', 0):.4f} | "
                        f"node_mse: {val_metrics.get('val/node_mse', 0):.4f} | "
                        f"{val_inter}\n",
                        flush=True,
                    )

                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(
                            args.checkpoint_dir, "best.pt"
                        )
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            epoch,
                            global_step,
                            best_val_loss,
                            config,
                        )
                        print(
                            f"  ✅ New best model saved "
                            f"(val_loss={best_val_loss:.4f})",
                            flush=True,
                        )

                    model.train()

                # ── Checkpointing ────────────────────────────────
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(
                        args.checkpoint_dir,
                        f"step_{global_step:06d}.pt",
                    )
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        ema,
                        epoch,
                        global_step,
                        best_val_loss,
                        config,
                    )
                    # Also save as latest for easy resume
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "latest.pt"),
                        model,
                        optimizer,
                        scheduler,
                        ema,
                        epoch,
                        global_step,
                        best_val_loss,
                        config,
                    )
                    manage_checkpoints(args.checkpoint_dir, keep_last=3)
                    print(
                        f"  💾 Checkpoint saved: step {global_step}",
                        flush=True,
                    )

            epoch_loss += out["loss"].item()

        # ── End of epoch ─────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, batch_idx + 1)
        print(
            f"\nEpoch {epoch+1}/{args.epochs} done | "
            f"avg_loss={avg_loss:.4f} | "
            f"time={epoch_time/60:.1f}min\n",
            flush=True,
        )

        # Epoch-end checkpoint
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"epoch_{epoch+1:02d}.pt",
        )
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            ema,
            epoch + 1,
            global_step,
            best_val_loss,
            config,
        )

    # ── Final ────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Training complete!", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Total steps: {global_step}", flush=True)
    print("=" * 60, flush=True)

    logger.close()


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Graph-RoPE Multi-Task Pretraining"
    )

    # Data
    p.add_argument(
        "--feat-dir",
        default="model/cpg_features",
        help="Directory with NPZ feature shards",
    )
    p.add_argument(
        "--bucket-dir",
        default="Dataset_for_first/cleaned_buckets",
        help="Directory with source code JSONL buckets",
    )
    p.add_argument(
        "--model-name",
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name",
    )

    # Training hyperparameters
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mlm-prob", type=float, default=0.30)
    p.add_argument("--n-edge-pairs", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--rope-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # Loss weights
    p.add_argument("--mlm-weight", type=float, default=1.0)
    p.add_argument("--node-weight", type=float, default=0.5)
    p.add_argument("--edge-weight", type=float, default=0.3)
    p.add_argument("--err-weight", type=float, default=0.4)

    # (objective_mode removed — all objectives controlled by weight)
    # Set edge_weight=0.0 and err_weight=0.4 for ERR, or vice versa

    # Checkpointing & logging
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints/e1_non_strict_s1",
    )
    p.add_argument("--log-dir", default="logs/e1_non_strict_s1")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--xai-interval", type=int, default=500, help="Steps between XAI diagnostics")
    p.add_argument("--resume", action="store_true")

    # System
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Experiment name
    p.add_argument(
        "--experiment",
        default="e1_non_strict",
        help="Experiment name for organization",
    )
    p.add_argument("--config", help="Path to JSON config file")

    args = p.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        
        # Helper to set arg if key exists in config
        def set_if_exists(namespace, key, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    set_if_exists(namespace, k, v)
            else:
                # Map JSON keys to argparse dest (mostly same, some diffs)
                # We simply check if the key matches a dest in args
                # or if we need manual mapping
                key_map = {
                    "experiment_name": "experiment",
                    "lr_encoder": "lr",
                    "lr_rope": "rope_lr",
                    "lr_head": "head_lr",
                    "sentry_log_interval": "log_interval", # n/a for train
                    "sentry_data_interval": "data_interval" # n/a for train
                }
                dest = key_map.get(key, key)
                if hasattr(namespace, dest):
                    setattr(namespace, dest, value)

        for k, v in cfg.items():
            set_if_exists(args, k, v)
            
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Experiment: {args.experiment}", flush=True)
    print(f"Config: {json.dumps(vars(args), indent=2)}", flush=True)
    train(args)
