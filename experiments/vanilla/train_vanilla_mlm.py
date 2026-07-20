#!/usr/bin/env python3
"""
Vanilla ModernBERT MLM Pretraining on Solidity.

Control baseline: same data, same epochs, same LR — just no Graph-RoPE.
Uses standard ModernBERT positional encoding (RoPE on sequential positions).
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

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, ModernBertForMaskedLM

# Add Implementation dir to path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPL_DIR = os.path.dirname(os.path.dirname(EXPERIMENTS_DIR))
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)

from models.shared.solidity_mlm_dataset import create_vanilla_dataloaders


# ═══════════════════════════════════════════════════════════════════
#  EMA
# ═══════════════════════════════════════════════════════════════════
class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, sd, device=None):
        self.shadow = sd["shadow"]
        self.decay = sd["decay"]
        if device:
            for k in self.shadow:
                self.shadow[k] = self.shadow[k].to(device)


# ═══════════════════════════════════════════════════════════════════
#  Logger
# ═══════════════════════════════════════════════════════════════════
class TrainingLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.csv_file = None
        self.csv_writer = None

    def log(self, step, metrics):
        if self.csv_writer is None:
            common = {"train/loss", "train/mlm_loss", "train/mlm_acc",
                      "train/perplexity", "train/grad_norm", "train/gpu_mem_gb",
                      "val/loss", "val/mlm_acc", "val/perplexity"}
            all_keys = sorted(list(common.union(metrics.keys())))
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, ["step"] + all_keys)
            self.csv_writer.writeheader()
            self.csv_file.flush()

        row = {"step": step}
        for k, v in metrics.items():
            if k in self.csv_writer.fieldnames:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                row[k] = f"{v:.6f}" if isinstance(v, float) else v
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint
# ═══════════════════════════════════════════════════════════════════
def save_checkpoint(path, model, optimizer, scheduler, ema, epoch, step, best_val, cfg):
    torch.save({
        "epoch": epoch, "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "best_val_loss": best_val, "config": cfg,
        "rng_states": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, ema):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"⚠️  Optimizer mismatch: {e}")
    try:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception as e:
        print(f"⚠️  Scheduler mismatch: {e}")
    if ema and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"], device=next(model.parameters()).device)
    if "rng_states" in ckpt:
        rng = ckpt["rng_states"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"]


def cleanup_old_step_checkpoints(ckpt_dir, keep=3):
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[1].split(".")[0]),
    )
    for f in ckpts[:-keep] if len(ckpts) > keep else []:
        os.remove(os.path.join(ckpt_dir, f))
        print(f"  Removed: {f}")


# ═══════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, val_loader, device, ema=None):
    model.eval()
    if ema:
        ema.apply(model)

    total_loss, total_acc, n = 0, 0, 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_labels)

        total_loss += out.loss.item()
        # Compute accuracy on masked tokens
        mask = mlm_labels != -100
        if mask.any():
            preds = out.logits[mask].argmax(-1)
            total_acc += (preds == mlm_labels[mask]).float().mean().item()
        n += 1

    if ema:
        ema.restore(model)

    return {
        "val/loss": total_loss / max(n, 1),
        "val/mlm_acc": total_acc / max(n, 1),
        "val/perplexity": math.exp(min(total_loss / max(n, 1), 20)),
    }


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {props.name}, Free: {free/1e9:.1f}GB / {total/1e9:.1f}GB")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Data
    print("Creating dataloaders...", flush=True)
    train_loader, val_loader, _ = create_vanilla_dataloaders(
        bucket_dir=args.bucket_dir,
        tokenizer=tokenizer,
        max_len=args.max_seq_len,
        batch_size=args.batch_size,
        mlm_prob=args.mlm_prob,
        max_contracts=args.max_contracts,
        num_workers=args.num_workers,
        seed=args.seed,
        npz_dir=args.npz_dir,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}", flush=True)

    # Model — vanilla ModernBERT with MLM head, NO Graph-RoPE
    print("Loading ModernBertForMaskedLM (vanilla)...", flush=True)
    model = ModernBertForMaskedLM.from_pretrained(args.model_name)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}", flush=True)

    # Optimizer — single LR (no separate rope/head groups)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # Scheduler: warmup + cosine
    warmup = LinearLR(optimizer, start_factor=1e-8/args.lr, end_factor=1.0,
                      total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])

    # EMA
    ema = EMAModel(model, decay=args.ema_decay)

    # Logger
    logger = TrainingLogger(args.log_dir)

    # Resume
    start_epoch, global_step, best_val = 0, 0, float("inf")
    resume_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path) and args.resume:
        start_epoch, global_step, best_val = load_checkpoint(
            resume_path, model, optimizer, scheduler, ema
        )
        print(f"Resumed: epoch={start_epoch}, step={global_step}, best_val={best_val:.4f}")

    # Save config
    cfg = vars(args)
    cfg["total_steps"] = total_steps
    cfg["steps_per_epoch"] = steps_per_epoch
    cfg["param_count"] = param_count
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    cleanup_old_step_checkpoints(args.checkpoint_dir)

    # Initial log
    logger.log(global_step, {
        "train/loss": 20.0, "train/mlm_loss": 20.0, "train/mlm_acc": 0.0,
        "train/perplexity": 0.0, "train/grad_norm": 0.0,
        "train/gpu_mem_gb": torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0,
    })

    # ── Training loop ─────────────────────────────────────────────
    print("=" * 60)
    print("Starting Vanilla ModernBERT Solidity MLM Pretraining...")
    print("=" * 60)

    start_time = time.time()
    start_step = global_step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_labels)
                loss = out.loss / args.grad_accum

            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(model)
                global_step += 1

                # Logging
                if global_step % args.log_every == 0:
                    # MLM accuracy
                    mask = mlm_labels != -100
                    mlm_acc = 0.0
                    if mask.any():
                        with torch.no_grad():
                            mlm_acc = (out.logits[mask].argmax(-1) == mlm_labels[mask]).float().mean().item()

                    mlm_loss = out.loss.item()
                    ppl = math.exp(min(mlm_loss, 20))
                    metrics = {
                        "train/loss": mlm_loss,
                        "train/mlm_loss": mlm_loss,
                        "train/mlm_acc": mlm_acc,
                        "train/perplexity": ppl,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/gpu_mem_gb": torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0,
                        "train/epoch": epoch,
                    }
                    elapsed = time.time() - start_time
                    steps_done = max(1, global_step - start_step)
                    eta_seconds = (elapsed / steps_done) * (total_steps - global_step)
                    eta_str = f"{int(eta_seconds // 3600):02d}h {int((eta_seconds % 3600) // 60):02d}m"

                    print(
                        f"  Step {global_step:>6d} | loss={mlm_loss:.4f} | "
                        f"acc={mlm_acc:.3f} | ppl={ppl:.1f} | "
                        f"gnorm={metrics['train/grad_norm']:.3f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                        f"ETA: {eta_str}",
                        flush=True,
                    )

                # Validation
                if global_step % args.val_every == 0:
                    val_metrics = validate(model, val_loader, device, ema)
                    logger.log(global_step, val_metrics)
                    print(
                        f"  📊 Val | loss={val_metrics['val/loss']:.4f} | "
                        f"acc={val_metrics['val/mlm_acc']:.3f} | "
                        f"ppl={val_metrics['val/perplexity']:.1f}",
                        flush=True,
                    )

                    if val_metrics["val/loss"] < best_val:
                        best_val = val_metrics["val/loss"]
                        save_checkpoint(
                            os.path.join(args.checkpoint_dir, "best.pt"),
                            model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg,
                        )
                        print(f"  ✅ New best: {best_val:.4f}", flush=True)

                # Step checkpoint
                if global_step % args.save_every == 0:
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, f"step_{global_step}.pt"),
                        model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg,
                    )
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "latest.pt"),
                        model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg,
                    )
                    cleanup_old_step_checkpoints(args.checkpoint_dir)

        # End of epoch
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} done in {epoch_time/60:.1f} min")
        print(f"{'='*60}\n")

        # Epoch checkpoint
        save_checkpoint(
            os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pt"),
            model, optimizer, scheduler, ema, epoch+1, global_step, best_val, cfg,
        )
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "latest.pt"),
            model, optimizer, scheduler, ema, epoch+1, global_step, best_val, cfg,
        )

    logger.close()
    print("Training complete!")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Flatten config into args namespace
    class Args:
        pass
    a = Args()

    a.bucket_dir = cfg["paths"]["bucket_dir"]
    a.checkpoint_dir = cfg["paths"]["checkpoint_dir"]
    a.log_dir = cfg["paths"]["log_dir"]
    a.npz_dir = cfg["paths"].get("npz_dir", "")

    # Model
    a.model_name = cfg["model"]["base_model"]
    a.max_seq_len = cfg["model"].get("max_seq_len", 2048)
    a.mlm_prob = cfg["model"].get("mlm_prob", 0.30)
    a.max_contracts = cfg["model"].get("max_contracts", 0)

    # Training
    t = cfg["training"]
    a.batch_size = t.get("batch_size", 16)
    a.grad_accum = t.get("grad_accum", 4)
    a.epochs = t.get("epochs", 10)
    a.lr = t.get("lr", 2e-5)
    a.weight_decay = t.get("weight_decay", 0.01)
    a.warmup_steps = t.get("warmup_steps", 2000)
    a.max_grad_norm = t.get("max_grad_norm", 0.5)
    a.ema_decay = t.get("ema_decay", 0.999)
    a.seed = t.get("seed", 42)
    a.num_workers = t.get("num_workers", 4)
    a.resume = t.get("resume", True)

    # Logging
    l = cfg["logging"]
    a.log_every = l.get("log_every", 100)
    a.val_every = l.get("val_every", 1000)
    a.save_every = l.get("save_every", 2000)

    train(a)


if __name__ == "__main__":
    main()
