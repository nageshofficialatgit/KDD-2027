#!/usr/bin/env python3
"""
Phase 4 Master Orchestrator (Robustness Evaluation)
Runs Obfuscation & Decompilation pipelines, then evaluates LLVT under multiple paradigms:
1. Whole Data (Mixed split)
2. Zero-Shot (Train ScrawlD -> Test SBCurated)
for both Clean and Obfuscated datasets.
"""
import os, subprocess, sys, json
from datetime import datetime

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, ".venv/bin/python3")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/scripts")
LLVT_SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/llvt_evaluator.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "Eval_Phase4")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_step(name, cmd, log_path):
    log(f"START: {name}")
    log(f"  CMD: {' '.join(cmd[:5])}...")
    log(f"  LOG: {log_path}")
    with open(log_path, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    if result.returncode != 0:
        log(f"  WARNING: {name} exited with code {result.returncode}")
    else:
        log(f"  DONE: {name}")
    return result.returncode

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "logs"), exist_ok=True)

    log("=" * 60)
    log("PHASE 4: OBFUSCATED CODE ROBUSTNESS EVALUATION")
    log("=" * 60)

    # ---- STEP 1: Tier A - Obfuscate contracts ----
    run_step("Tier A: Obfuscate Contracts",
             [sys.executable, os.path.join(SCRIPTS_DIR, "obfuscate_contracts.py")],
             os.path.join(LOG_DIR, "logs", "step1_obfuscate.log"))

    # ---- STEP 2: Tier B - Compile & Decompile contracts ----
    run_step("Tier B: Compile & Decompile",
             [sys.executable, os.path.join(SCRIPTS_DIR, "compile_and_decompile.py")],
             os.path.join(LOG_DIR, "logs", "step2_decompile.log"))

    # Annotation Paths
    clean_anno = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/verified_sbcurated_ground_truth.jsonl")
    obf_anno = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/obfuscated/obfuscated_annotations.jsonl")

    # ---- STEP 3: LLVT CLEAN (Whole Data) ----
    run_step("LLVT Clean (Whole Data)",
             [PYTHON, LLVT_SCRIPT,
              "--model_name", "modernbert_clean_whole",
              "--model_type", "modernbert",
              "--base_model_path", "answerdotai/ModernBERT-base",
              "--max_len", "1024",
              "--batch_size", "2",
              "--epochs", "10",
              "--lr", "2e-5",
              "--annotation_file", clean_anno],
             os.path.join(LOG_DIR, "logs", "step3_llvt_clean_whole.log"))

    # ---- STEP 4: LLVT CLEAN (Zero-Shot) ----
    run_step("LLVT Clean (Zero-Shot)",
             [PYTHON, LLVT_SCRIPT,
              "--model_name", "modernbert_clean_zeroshot",
              "--model_type", "modernbert",
              "--base_model_path", "answerdotai/ModernBERT-base",
              "--max_len", "1024",
              "--batch_size", "2",
              "--epochs", "10",
              "--lr", "2e-5",
              "--annotation_file", clean_anno,
              "--zero_shot"],
             os.path.join(LOG_DIR, "logs", "step4_llvt_clean_zeroshot.log"))

    # ---- STEP 5: LLVT OBFUSCATED (Whole Data) ----
    run_step("LLVT Obfuscated (Whole Data)",
             [PYTHON, LLVT_SCRIPT,
              "--model_name", "modernbert_obf_whole",
              "--model_type", "modernbert",
              "--base_model_path", "answerdotai/ModernBERT-base",
              "--max_len", "1024",
              "--batch_size", "2",
              "--epochs", "10",
              "--lr", "2e-5",
              "--annotation_file", obf_anno],
             os.path.join(LOG_DIR, "logs", "step5_llvt_obf_whole.log"))

    # ---- STEP 6: LLVT OBFUSCATED (Zero-Shot) ----
    run_step("LLVT Obfuscated (Zero-Shot)",
             [PYTHON, LLVT_SCRIPT,
              "--model_name", "modernbert_obf_zeroshot",
              "--model_type", "modernbert",
              "--base_model_path", "answerdotai/ModernBERT-base",
              "--max_len", "1024",
              "--batch_size", "2",
              "--epochs", "10",
              "--lr", "2e-5",
              "--annotation_file", obf_anno,
              "--zero_shot"],
             os.path.join(LOG_DIR, "logs", "step6_llvt_obf_zeroshot.log"))

    log("=" * 60)
    log("PHASE 4 COMPLETE")
    log("=" * 60)

if __name__ == "__main__":
    main()
