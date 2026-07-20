#!/usr/bin/env python3
"""
Run LLVT on all WIRE ablation variants in text-only mode (no graph features).
Uses pretrained checkpoints from ./outputs/checkpoints/.
Saves all outputs to the outputs directory.
"""

import os
import subprocess
from datetime import datetime

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("DATA_ROOT", "./outputs")
EVAL_DIR = os.path.join(DATA_ROOT, "Eval_Phase2")
LOG_DIR = os.path.join(EVAL_DIR, "logs")
PYTHON = os.path.join(PROJECT_ROOT, ".venv/bin/python3")
SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/llvt_evaluator.py")
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints")

# All wire_abl variants
MODELS = [
    {
        "name": "wire_abl_text_only_llvt",
        "checkpoint": os.path.join(CKPT_DIR, "wire_abl", "best.pt"),
    },
    {
        "name": "wire_abl_8d_text_only_llvt",
        "checkpoint": os.path.join(CKPT_DIR, "wire_abl_8d", "best.pt"),
    },
    {
        "name": "wire_abl_10d_text_only_llvt",
        "checkpoint": os.path.join(CKPT_DIR, "wire_abl_10d", "best.pt"),
    },
    {
        "name": "wire_abl_2048_text_only_llvt",
        "checkpoint": os.path.join(CKPT_DIR, "wire_abl_2048", "best.pt"),
    },
]

def run(cmd, log_path, env):
    print(f"[{datetime.now()}] Running: {' '.join(cmd)}")
    print(f"  Log: {log_path}")
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    for m in MODELS:
        if not os.path.exists(m["checkpoint"]):
            print(f"SKIP {m['name']} (checkpoint not found: {m['checkpoint']})")
            continue

        log_path = os.path.join(LOG_DIR, f"{m['name']}.log")

        # Skip if already completed
        if os.path.exists(log_path):
            try:
                content = open(log_path, errors="ignore").read()
                if "LLVT Localization Results" in content:
                    print(f"SKIP {m['name']} (already complete)")
                    continue
            except:
                pass

        cmd = [
            PYTHON, SCRIPT,
            "--model_name", m["name"],
            "--model_type", "wire",
            "--base_model_path", "answerdotai/ModernBERT-base",
            "--max_len", "1024",
            "--batch_size", "2",
            "--epochs", "10",
            "--lr", "2e-5",
            "--num_gat_layers", "1",
            "--dropout", "0.1",
            "--contrastive_weight", "0.1",
            "--checkpoint", m["checkpoint"],
        ]

        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        # Redirect temp files to outputs directory
        env["TMPDIR"] = os.path.join(DATA_ROOT, "tmp")
        os.makedirs(env["TMPDIR"], exist_ok=True)

        try:
            run(cmd, log_path, env)
            print(f"  Done: {m['name']}")
        except Exception as e:
            print(f"  ERROR {m['name']}: {e}")

    print(f"\n--- LLVT Text-Only Complete (all wire_abl variants) ---")

if __name__ == "__main__":
    main()
