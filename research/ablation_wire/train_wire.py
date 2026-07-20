#!/usr/bin/env python3
"""
WIRE Standalone Ablation Training Script.
"""

import os
import sys
import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure we can import our models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.wire.wire_rope_model import WIREModernBERT, WIREConfig
from models.graph_rope_dataset import create_dataloaders

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_wire.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])

    # ── Dataloader ──
    train_loader, val_loader, _ = create_dataloaders(
        npz_dir=cfg["paths"]["feat_dir"],
        bucket_dir=cfg["paths"]["bucket_dir"],
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
    )

    # ── Model ──
    wire_config = WIREConfig(
        d_head=768 // 12, 
        n_features=cfg["model"].get("n_features", 6)
    )
    model = WIREModernBERT(
        model_name=cfg["model"]["base_model"],
        wire_config=wire_config,
        mlm_weight=cfg["model"]["mlm_weight"],
        err_weight=cfg["model"]["err_weight"],
    ).to(device)

    # ── Optimizer ──
    # Corrected step calculation for grad accumulation
    steps_per_epoch = len(train_loader) // cfg["training"]["grad_accum"]
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    
    param_groups = model.get_param_groups(
        encoder_lr=cfg["training"]["lr_encoder"],
        wire_lr=cfg["training"]["lr_rope"],
        head_lr=cfg["training"]["lr_head"]
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg["training"]["warmup_steps"], total_steps)

    print(f"Starting WIRE Ablation: {total_steps} steps total.")

    start_epoch = 0
    step = 0
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    
    ckpt_to_load = latest_ckpt if os.path.exists(latest_ckpt) else (best_ckpt if os.path.exists(best_ckpt) else None)
    if ckpt_to_load:
        print(f"Loading checkpoint from {ckpt_to_load}...")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"]
            step = ckpt["step"]
            # If loaded best.pt (end of an epoch), increment epoch
            if ckpt_to_load == best_ckpt:
                start_epoch += 1
            print(f"Resumed from epoch {start_epoch}, step {step}")
        else:
            model.load_state_dict(ckpt)
            print("Loaded raw model state dict.")
            # Hardcode resume for the frozen 8D run
            if "wire_abl_8d" in ckpt_dir:
                start_epoch = 2
                step = start_epoch * steps_per_epoch
                print(f"Hardcoding resume for 8D run at epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"] / cfg["training"]["grad_accum"]
            loss.backward()

            if (i + 1) % cfg["training"]["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                
                if step % 100 == 0:
                    mlm_acc = outputs.get("mlm_acc", 0.0)
                    pbar.set_postfix({"loss": outputs["loss"].item(), "mlm_acc": mlm_acc.item() if isinstance(mlm_acc, torch.Tensor) else mlm_acc})

                # Periodic checkpoint every 500 optimizer steps
                if step % 500 == 0 and step > 0:
                    ckpt_dir = cfg["paths"]["checkpoint_dir"]
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                    }, ckpt_path)
                    print(f"\n[Checkpoint] Saved step {step} to {ckpt_path}")

        # Save best checkpoint after each epoch
        ckpt_dir = cfg["paths"]["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"best.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        }, ckpt_path)
        print(f"Saved epoch {epoch} checkpoint to {ckpt_path}")

    print("Training Complete.")

if __name__ == "__main__":
    train()
