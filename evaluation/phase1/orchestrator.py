#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime

# Define paths
ROOT_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT_DIR, "Eval")
LOGS_DIR = os.path.join(EVAL_DIR, "logs")

EVAL_SCRIPT = os.path.join(ROOT_DIR, "Implementation", "evaluation", "phase1", "model_eval.py")
PREP_SCRIPT = os.path.join(ROOT_DIR, "Implementation", "evaluation", "phase1", "prep_graphcodebert.py")
PYTHON_EX = os.path.join(ROOT_DIR, ".venv", "bin", "python3")

# Define Models to run
MODELS = [
    # {"name": "modernbert", "type": "modernbert", "checkpoint": None},
    # {"name": "codebert", "type": "codebert", "checkpoint": None},
    # {"name": "graphcodebert", "type": "graphcodebert", "checkpoint": None},
    # {"name": "e1_err", "type": "wire", "checkpoint": os.path.join(ROOT_DIR, "checkpoints", "e1_err", "best.pt")},
    {"name": "wire_abl", "type": "wire", "checkpoint": os.path.join(ROOT_DIR, "checkpoints", "wire_abl", "best.pt")},
    {"name": "wire_abl_comment", "type": "wire", "checkpoint": os.path.join(ROOT_DIR, "checkpoints", "wire_abl_comment", "best.pt")},
    {"name": "exe_mlm_v5", "type": "wire", "checkpoint": os.path.join(ROOT_DIR, "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")}
]


def run_cmd(cmd, log_path, desc):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ==> Starting {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")
    
    with open(log_path, 'w') as f:
        f.write(f"--- STARTING {desc} ---\nCommand: {' '.join(cmd)}\n\n")
        f.flush()
        
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        try:
            # Wait for process to complete
            process.wait()
            if process.returncode == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] {desc} completed.")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {desc} failed with code {process.returncode}. Check logs.")
        except KeyboardInterrupt:
            process.terminate()
            print("\nProcess interrupted by user.")
            sys.exit(1)

def orchestrate():
    print("=============================================")
    print(" Phase 1: Comprehensive Evaluation Orchestrator ")
    print("=============================================")

    # 1. Check if GraphCodeBERT needs data flow extraction
    print("\n--- Pre-flight Checks ---")
    gcb_cache_flag = os.path.join(ROOT_DIR, "Implementation", "evaluation", "cache", "graphs", ".gcb_ready")
    if not os.path.exists(gcb_cache_flag):
        print("GraphCodeBERT data flow not found. Initiating extraction...")
        prep_log = os.path.join(LOGS_DIR, "prep_graphcodebert.log")
        run_cmd([PYTHON_EX, PREP_SCRIPT], prep_log, "GraphCodeBERT Preprocessor")
        
        # Touch flag file if successful
        if os.path.exists(prep_log) and "SUCCESS" in open(prep_log).read(): # Naive check
             with open(gcb_cache_flag, 'w') as f: f.write("done")
    else:
        print("GraphCodeBERT data flow already cached.")

    # 2. Run sequential evaluation
    for m in MODELS:
        print(f"\n--- Queuing Model: {m['name']} ---")
        
        # Validate checkpoint if specified
        if m["checkpoint"] and not os.path.exists(m["checkpoint"]):
            print(f"[SKIP] Checkpoint missing: {m['checkpoint']}")
            continue
            
        log_path = os.path.join(LOGS_DIR, f"{m['name']}_eval.log")
        cmd = [
            PYTHON_EX, EVAL_SCRIPT,
            "--model_type", m["type"],
            "--model_name", m["name"]
        ]
        
        if m["checkpoint"]:
            cmd.extend(["--checkpoint", m["checkpoint"]])
            
        try:
            run_cmd(cmd, log_path, f"Evaluation: {m['name']}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [CRITICAL ERROR] Failed to execute {m['name']}: {e}")
            with open(log_path, 'a') as f:
                f.write(f"\n\n[CRITICAL ERROR] Failed to execute: {e}\n")
        
        # Small cooldown between heavy GPU tasks
        time.sleep(5)
        
    print("\n=============================================")
    print(" Orchestration Complete. ")
    print("=============================================")
    
    # 3. Trigger final report generation
    report_script = os.path.join(ROOT_DIR, "Implementation", "evaluation", "phase1", "report_generator.py")
    report_log = os.path.join(LOGS_DIR, "report_generation.log")
    run_cmd([PYTHON_EX, report_script], report_log, "Final Report Generation")


if __name__ == "__main__":
    orchestrate()
