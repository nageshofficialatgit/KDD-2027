#!/usr/bin/env python3
"""
WIRE Standalone Ablation Training Script.

Full-featured training with:
  - bf16 mixed precision (A100 native)
  - Checkpointing with full RNG state recovery
  - CSV + TensorBoard logging
  - EMA (Exponential Moving Average) model
  - Gradient clipping and warm-up + cosine decay
  - Validation loop
  - GPU memory monitoring
  - Instability alerts
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
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

# Add Implementation dir (parent of experiments/) to path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.dirname(os.path.dirname(EXPERIMENTS_DIR))
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)

from models.metp.metp_model import METPModernBERT
from models.wire.wire_rope import WIREConfig
from models.metp.metp_dataset import create_dataloaders

print(">>> BTP_TELEMETRY: train_metp.py loaded (full-featured)")


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
                print("TensorBoard not available, using CSV only", flush=True)

    def log(self, step: int, metrics: dict):
        """Log metrics at a given step."""
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
                if k in self.csv_writer.fieldnames:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    row[k] = f"{v:.6f}" if isinstance(v, float) else v
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, torch.Tensor)):
                    self.tb_writer.add_scalar(k, v, step)

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()


# ═════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═════════════════════════════════════════════════════════════════════════
def check_disk_space(path, min_gb=5.0):
    """Check if sufficient disk space is available before saving."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            print(
                f"  ⚠️  LOW DISK: {free_gb:.1f}GB free "
                f"(need {min_gb}GB)",
                flush=True,
            )
            return False
        return True
    except Exception:
        return True  # If we can't check, proceed anyway


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
    # Atomic save: write to temp, then rename
    tmp_path = path + ".tmp"
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
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)  # Atomic rename


def load_checkpoint(path: str, model, optimizer, scheduler, ema):
    """Load checkpoint and restore full training state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"], device=next(model.parameters()).device)

    # Restore RNG states
    rng = ckpt["rng_states"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"])

    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"]


def cleanup_old_step_checkpoints(ckpt_dir: str):
    """Remove all step_*.pt files — only latest.pt + best.pt are kept."""
    for f in os.listdir(ckpt_dir):
        if f.startswith("step_") and f.endswith(".pt"):
            path = os.path.join(ckpt_dir, f)
            os.remove(path)
            print(f"  🗑️  Removed old checkpoint: {f}", flush=True)


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
    total_edge_loss = 0
    total_mlm_acc = 0
    total_edge_acc = 0
    n_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast('cuda', dtype=torch.bfloat16):
            out = model(**batch)

        total_loss += out["loss"].item()
        if "mlm_loss" in out:
            total_mlm_loss += out["mlm_loss"].item()
        if "edge_loss" in out:
            total_edge_loss += out["edge_loss"].item()
        if "edge_acc" in out:
            total_edge_acc += out["edge_acc"].item()
        if "mlm_acc" in out:
            total_mlm_acc += out["mlm_acc"].item()
        n_batches += 1

    if use_ema and ema:
        ema.restore(model)

    if n_batches == 0:
        return {}

    return {
        "val/loss": total_loss / n_batches,
        "val/mlm_loss": total_mlm_loss / n_batches,
        "val/edge_loss": total_edge_loss / n_batches,
        "val/mlm_acc": total_mlm_acc / n_batches,
        "val/edge_acc": total_edge_acc / n_batches,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═════════════════════════════════════════════════════════════════════════
def train():
    """Main training function."""
    parser = argparse.ArgumentParser(description="METP Ablation Training")
    parser.add_argument("--config", type=str, default="configs/metp/config_metp.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

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

    # Seed everything
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Directories
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    log_dir = cfg["paths"]["log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])

    # ── Data ─────────────────────────────────────────────────────────
    print("Creating dataloaders...", flush=True)
    train_loader, val_loader, _ = create_dataloaders(
        npz_dir=cfg["paths"]["feat_dir"],
        bucket_dir=cfg["paths"]["bucket_dir"],
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
        seed=seed,
    )

    grad_accum = cfg["training"]["grad_accum"]
    epochs = cfg["training"]["epochs"]
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    print(
        f"Steps per epoch: {steps_per_epoch}, "
        f"Total steps: {total_steps}",
        flush=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    print("Initializing METP model...", flush=True)
    wire_config = WIREConfig(d_head=768 // 12)
    model = METPModernBERT(
        model_name=cfg["model"]["base_model"],
        wire_config=wire_config,
        mlm_weight=cfg["model"]["mlm_weight"],
        edge_weight=cfg["model"].get("edge_weight", 0.5), # Default to 0.5 for METP
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Parameters: {param_count:,} total, "
        f"{trainable_count:,} trainable",
        flush=True,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    param_groups = model.get_param_groups(
        encoder_lr=cfg["training"]["lr_encoder"],
        wire_lr=cfg["training"]["lr_rope"],
        head_lr=cfg["training"]["lr_head"],
    )
    optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)

    # ── Scheduler: linear warmup + cosine decay ──────────────────────
    warmup_steps = cfg["training"]["warmup_steps"]
    lr_encoder = cfg["training"]["lr_encoder"]
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8 / lr_encoder,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # ── EMA ──────────────────────────────────────────────────────────
    ema_decay = cfg["training"].get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)

    # ── Logger ───────────────────────────────────────────────────────
    logger = TrainingLogger(log_dir)

    # ── Resume from checkpoint ───────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    resume_path = os.path.join(ckpt_dir, "latest.pt")
    resume = cfg["training"].get("resume", True)
    if os.path.exists(resume_path) and resume:
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
    cfg["_computed"] = {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "param_count": param_count,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── Logging intervals ────────────────────────────────────────────
    log_every = cfg.get("logging", {}).get("log_every", 100)
    val_every = cfg.get("logging", {}).get("val_every", 1000)
    save_every = cfg.get("logging", {}).get("save_every", 2000)
    max_grad_norm = cfg["training"]["max_grad_norm"]
    training_start_time = time.time()

    # ── Clean up any leftover step checkpoints from previous runs ────
    cleanup_old_step_checkpoints(ckpt_dir)

    # ── Initial log ──────────────────────────────────────────────────
    logger.log(global_step, {
        "train/loss": 20.0,
        "train/mlm_loss": 20.0,
        "train/edge_loss": 0.0,
        "train/mlm_acc": 0.0,
        "train/edge_acc": 0.0,
        "train/perplexity": 0.0,
        "train/grad_norm": 0.0,
        "train/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "train/wall_time_min": 0.0,
    })

    # ── Training ─────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("Starting METP Ablation Training...", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward + loss
            with autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch)
                loss = out["loss"] / grad_accum

            # Backward
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                ema.update(model)
                global_step += 1

                # ── Logging ──────────────────────────────────────
                if global_step % log_every == 0:
                    gpu_mem = (
                        torch.cuda.memory_allocated() / 1e9
                        if torch.cuda.is_available()
                        else 0
                    )
                    gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    wall_min = (time.time() - training_start_time) / 60.0

                    metrics = {
                        "train/loss": out["loss"].item(),
                        "train/grad_norm": gn,
                        "train/lr_encoder": optimizer.param_groups[0]["lr"],
                        "train/lr_wire": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0,
                        "train/gpu_mem_gb": gpu_mem,
                        "train/epoch": epoch,
                        "train/wall_time_min": wall_min,
                    }
                    if "mlm_loss" in out:
                        mlm_val = out["mlm_loss"].item()
                        metrics["train/mlm_loss"] = mlm_val
                        metrics["train/perplexity"] = min(math.exp(mlm_val), 1e6)
                    if "edge_loss" in out:
                        metrics["train/edge_loss"] = out["edge_loss"].item()
                    if "edge_acc" in out:
                        metrics["train/edge_acc"] = out["edge_acc"].item()
                    if "mlm_acc" in out:
                        metrics["train/mlm_acc"] = out["mlm_acc"].item()

                    logger.log(global_step, metrics)

                    # ETA estimation
                    steps_done = max(global_step - (start_epoch * steps_per_epoch), 1)
                    eta_min = wall_min / steps_done * (total_steps - global_step)
                    eta_str = f"{eta_min/60:.1f}h" if eta_min > 60 else f"{eta_min:.0f}m"

                    # Build loss string
                    parts = [f"loss={out['loss'].item():.4f}"]
                    if "mlm_loss" in out:
                        parts.append(f"mlm={out['mlm_loss'].item():.4f}")
                    if "edge_loss" in out:
                        parts.append(f"edge={out['edge_loss'].item():.4f}")
                    if "mlm_acc" in out:
                        parts.append(f"acc={out['mlm_acc'].item():.3f}")
                    if "edge_acc" in out:
                        parts.append(f"e_acc={out['edge_acc'].item():.3f}")
                    parts.append(f"grad={gn:.2f}")
                    parts.append(f"GPU={gpu_mem:.1f}GB")
                    parts.append(f"ETA={eta_str}")

                    print(
                        f"  [{epoch+1}/{epochs}] "
                        f"step {global_step}/{total_steps} | "
                        + " | ".join(parts),
                        flush=True,
                    )

                    # Alert on instability
                    if gn > 10.0:
                        print(
                            f"  ⚠️  HIGH GRADIENT NORM: {gn:.2f} "
                            f"(threshold: 10.0)",
                            flush=True,
                        )

                # ── Validation ────────────────────────────────────
                if global_step % val_every == 0:
                    print(f"\n  📊 Validation at step {global_step}...", flush=True)
                    val_metrics = validate(model, val_loader, device, use_ema=True, ema=ema)
                    if val_metrics:
                        logger.log(global_step, val_metrics)
                        vl = val_metrics.get("val/loss", float("inf"))
                        print(
                            f"  val/loss={vl:.4f} | "
                            f"val/mlm={val_metrics.get('val/mlm_loss', 0):.4f} | "
                            f"val/edge={val_metrics.get('val/edge_loss', 0):.4f} | "
                            f"val/acc={val_metrics.get('val/mlm_acc', 0):.3f} | "
                            f"val/e_acc={val_metrics.get('val/edge_acc', 0):.3f}",
                            flush=True,
                        )

                        # Save best
                        if vl < best_val_loss:
                            best_val_loss = vl
                            if check_disk_space(ckpt_dir):
                                best_path = os.path.join(ckpt_dir, "best.pt")
                                save_checkpoint(
                                    best_path, model, optimizer, scheduler,
                                    ema, epoch, global_step, best_val_loss, cfg,
                                )
                                print(f"  ✅ New best: {vl:.4f}", flush=True)
                            else:
                                print(f"  ✅ New best: {vl:.4f} (skipped save — low disk)", flush=True)

                    model.train()

                # ── Checkpointing (only latest.pt) ────────────────
                if global_step % save_every == 0:
                    if check_disk_space(ckpt_dir):
                        latest_path = os.path.join(ckpt_dir, "latest.pt")
                        save_checkpoint(
                            latest_path, model, optimizer, scheduler,
                            ema, epoch, global_step, best_val_loss, cfg,
                        )
                        print(f"  💾 Checkpoint saved: latest.pt (step {global_step})", flush=True)
                    else:
                        print(f"  ⚠️  Checkpoint SKIPPED (low disk space)", flush=True)

        # End of epoch
        elapsed = time.time() - epoch_start
        print(
            f"\n  Epoch {epoch+1}/{epochs} complete in {elapsed:.0f}s\n",
            flush=True,
        )

    # ── Final save ───────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(
        final_path, model, optimizer, scheduler,
        ema, epochs, global_step, best_val_loss, cfg,
    )
    logger.close()
    print("Training Complete.", flush=True)


if __name__ == "__main__":
    train()
