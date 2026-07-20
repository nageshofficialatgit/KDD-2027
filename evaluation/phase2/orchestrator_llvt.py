#!/usr/bin/env python3
"""
LLVT Orchestrator: Run all models through line-level vulnerability tagging.
Includes all encoder variants: ModernBERT, CodeBERT, GraphCodeBERT, WIRE(E1-ERR), WIRE-Abl, WIRE-Abl-Comment.
"""

import os
import subprocess
from datetime import datetime

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "Eval_Phase2")
LOG_DIR = os.path.join(EVAL_DIR, "logs")
PYTHON = os.path.join(PROJECT_ROOT, ".venv/bin/python3")
SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/llvt_evaluator.py")

NPZ_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards")
DF_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/graphcodebert_df")

MODELS = [
    # 1. ModernBERT (no graph features)
    {
        "name": "modernbert_llvt",
        "type": "modernbert",
        "base": "answerdotai/ModernBERT-base",
        "max_len": 1024,
        "epochs": 10,
    },
    # 2. CodeBERT
    {
        "name": "codebert_llvt",
        "type": "codebert",
        "base": "microsoft/codebert-base",
        "max_len": 512,
        "epochs": 10,
    },
    # 3. GraphCodeBERT (with data flow)
    {
        "name": "graphcodebert_llvt",
        "type": "graphcodebert",
        "base": "microsoft/graphcodebert-base",
        "max_len": 512,
        "df_dir": DF_DIR,
        "epochs": 10,
    },
    # 4. WIRE E1-ERR (with graph features)
    {
        "name": "e1_err_llvt",
        "type": "wire",
        "base": "answerdotai/ModernBERT-base",
        "max_len": 1024,
        "npz_dir": NPZ_DIR,
        "epochs": 10,
    },
    # 5. WIRE Ablation
    {
        "name": "wire_abl_llvt",
        "type": "wire",
        "base": "answerdotai/ModernBERT-base",
        "max_len": 1024,
        "npz_dir": NPZ_DIR,
        "epochs": 10,
    },
    # 6. WIRE Ablation Comment
    {
        "name": "wire_abl_comment_llvt",
        "type": "wire",
        "base": "answerdotai/ModernBERT-base",
        "max_len": 1024,
        "npz_dir": NPZ_DIR,
        "epochs": 10,
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
            "--model_type", m["type"],
            "--base_model_path", m["base"],
            "--max_len", str(m["max_len"]),
            "--batch_size", "2",
            "--epochs", str(m.get("epochs", 10)),
            "--lr", "2e-5",
            "--num_gat_layers", "1",
            "--dropout", "0.1",
            "--contrastive_weight", "0.1",
            "--annotation_file", os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/verified_sbcurated_ground_truth.jsonl"),
        ]

        if "npz_dir" in m:
            cmd.extend(["--npz_dir", m["npz_dir"]])
        if "df_dir" in m:
            cmd.extend(["--df_dir", m["df_dir"]])

        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        try:
            run(cmd, log_path, env)
            print(f"  Done: {m['name']}")
        except Exception as e:
            print(f"  ERROR {m['name']}: {e}")

    print(f"\n--- LLVT Orchestration Complete ---")


if __name__ == "__main__":
    main()
