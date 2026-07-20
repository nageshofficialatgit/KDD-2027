#!/usr/bin/env python3
import os
import subprocess
import argparse
import time
from datetime import datetime

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
PYTHON_BIN = os.path.join(PROJECT_ROOT, ".venv/bin/python3")

# Evaluation Scripts
TIER1_SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/benchmarks/tier1_detection/benchmark_supervised.py")
TIER2_SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/graph_mil_evaluator.py")
TIER2_LLVT_SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/llvt_evaluator.py")

# Output/Logs
LOG_DIR = os.path.join(PROJECT_ROOT, "logs/exe_graph_mlm_pretrain/master_eval")
os.makedirs(LOG_DIR, exist_ok=True)

def run_cmd(cmd, desc, log_name):
    log_path = os.path.join(LOG_DIR, f"{log_name}.log")
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {desc}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Logging to: {log_path}")
    
    start_time = time.time()
    with open(log_path, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("\nInterrupted by user.")
            return False
            
    duration = (time.time() - start_time) / 60.0
    if process.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: {desc} finished in {duration:.2f} mins.")
        return True
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] FAILED: {desc} exited with code {process.returncode}.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Orchestrator for Exe-MLM Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the pretrained checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default="exe_mlm_new_strategy", help="Base name for the model in reports")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size for quick benchmarking")
    args = parser.parse_args()

    checkpoint_abs = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_abs):
        print(f"ERROR: Checkpoint not found at {checkpoint_abs}")
        return

    print("==========================================================")
    print(f" EXE-MLM COMPREHENSIVE EVALUATION PROCESS ")
    print(f" Model: {args.model_name}")
    print(f" Checkpoint: {checkpoint_abs}")
    print("==========================================================")

    # --- Phase 1: Tier 1 Detection ---
    t1_cmd = [
        PYTHON_BIN, TIER1_SCRIPT,
        "--model_type", "wire",
        "--mode", "finetune",
        "--checkpoint", checkpoint_abs,
        "--epochs", "5",
        "--batch_size", "8",
        "--lr", "5e-5",
        "--output_suffix", f"{args.model_name}_t1_high"
    ]
    if args.limit > 0:
        t1_cmd.extend(["--limit", str(args.limit)])
        
    run_cmd(t1_cmd, "Tier 1: Supervised Detection", f"{args.model_name}_tier1")

    # --- Phase 2: Tier 2 Localization ---
    t2_cmd = [
        PYTHON_BIN, TIER2_SCRIPT,
        "--model_name", f"{args.model_name}_t2",
        "--model_type", "wire",
        "--base_model_path", "answerdotai/ModernBERT-base",
        "--checkpoint", checkpoint_abs,
        "--npz_dir", os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards"),
        "--epochs", "3",
        "--lr", "2e-5",
    ]
    if args.limit > 0:
        t2_cmd.extend(["--limit", str(args.limit)])
    else:
        # Default limit for localization if not specified, to prevent infinite runs
        t2_cmd.extend(["--limit", "1000"])

    run_cmd(t2_cmd, "Tier 2: Graph-Aware Localization (MIL)", f"{args.model_name}_tier2_mil")

    # --- Phase 3: Tier 2 High-Precision Localization (LLVT) ---
    llvt_cmd = [
        PYTHON_BIN, TIER2_LLVT_SCRIPT,
        "--model_name", f"{args.model_name}_llvt",
        "--model_type", "wire",
        "--base_model_path", "answerdotai/ModernBERT-base",
        "--checkpoint", checkpoint_abs,
        "--npz_dir", os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards"),
        "--epochs", "5",
        "--lr", "2e-5",
    ]
    run_cmd(llvt_cmd, "Tier 2: High-Precision Localization (LLVT)", f"{args.model_name}_tier2_llvt")

    print("\n==========================================================")
    print(" ALL EVALUATION PHASES COMPLETE ")
    print(f" Results available in: {PROJECT_ROOT}/Implementation/evaluation/results/")
    print("==========================================================")

if __name__ == "__main__":
    main()
