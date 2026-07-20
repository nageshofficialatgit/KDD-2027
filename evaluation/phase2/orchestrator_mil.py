import os
import subprocess
import time
from datetime import datetime

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
EVAL_PHASE2_DIR = os.path.join(PROJECT_ROOT, "Eval_Phase2")
LOG_DIR = os.path.join(EVAL_PHASE2_DIR, "logs")
PYTHON_BIN = os.path.join(PROJECT_ROOT, ".venv/bin/python3")
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "Implementation/evaluation/phase2/mil_evaluator.py")

NPZ_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/eval_shards")
DF_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/cache/graphs/graphcodebert_df")

MODELS = [
    {"name": "modernbert_mil", "type": "modernbert", "base": "answerdotai/ModernBERT-base", "max_len": 1024},
    {"name": "codebert_mil", "type": "codebert", "base": "microsoft/codebert-base", "max_len": 512},
    {"name": "graphcodebert_mil", "type": "graphcodebert", "base": "microsoft/graphcodebert-base", "max_len": 512, "df_dir": DF_DIR},
    {"name": "e1_err_mil", "type": "wire", "base": "answerdotai/ModernBERT-base", "max_len": 1024, "npz_dir": NPZ_DIR},
    {"name": "wire_abl_mil", "type": "wire", "base": "answerdotai/ModernBERT-base", "max_len": 1024, "npz_dir": NPZ_DIR},
    {"name": "wire_abl_comment_mil", "type": "wire", "base": "answerdotai/ModernBERT-base", "max_len": 1024, "npz_dir": NPZ_DIR},
]

def run_cmd(cmd, log_path, env):
    print(f"[{datetime.now()}] Executing: {' '.join(cmd)}")
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

def orchestrate():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    for m in MODELS:
        log_path = os.path.join(LOG_DIR, f"{m['name']}.log")
        
        # Check if already completed
        if os.path.exists(log_path) and "Localization Results" in open(log_path, errors="ignore").read():
            print(f"Skipping {m['name']} (already complete)")
            continue
            
        cmd = [
            PYTHON_BIN, EVAL_SCRIPT,
            "--model_name", m["name"],
            "--model_type", m["type"],
            "--base_model_path", m["base"],
            "--max_len", str(m["max_len"]),
            "--batch_size", "2",
            "--epochs", "3",
            "--lr", "2e-5"
        ]
        
        if "npz_dir" in m:
            cmd.extend(["--npz_dir", m["npz_dir"]])
        if "df_dir" in m:
            cmd.extend(["--df_dir", m["df_dir"]])
            
        env = os.environ.copy()
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        
        try:
            run_cmd(cmd, log_path, env)
        except Exception as e:
            print(f"Error running {m['name']}: {e}")
            
    print("--- Phase 2 MIL Orchestration Complete ---")

if __name__ == "__main__":
    orchestrate()
