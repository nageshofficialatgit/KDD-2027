#!/usr/bin/env python3
import os
import json
import sys
from pathlib import Path

# Setup paths to import from data_prep
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__))))
DATA_PREP_DIR = PROJECT_ROOT / "Implementation" / "data_prep"
sys.path.append(str(DATA_PREP_DIR))

from clean_dataset import clean_source
from solc_resolver import detect_pragma_version, preflight_install, get_all_installed

# Input/Output paths
LABELS_PATH = PROJECT_ROOT / "Implementation" / "evaluation" / "registry" / "labels.jsonl"
EVAL_BUCKET_DIR = PROJECT_ROOT / "Dataset_for_first" / "eval_buckets"
OUTPUT_DIR = PROJECT_ROOT / "Implementation" / "evaluation" / "cache" / "graphs" / "eval_shards"
PYTHON_BIN = "python3"

def prepare_eval_jsonl():
    print(f"Preparing eval buckets in {EVAL_BUCKET_DIR}...")
    os.makedirs(EVAL_BUCKET_DIR, exist_ok=True)
    
    bucket_counts = {}
    bucket_handles = {}
    
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            source = item.get("source_code", "")
            filename = item.get("filename")
            
            # Clean and detect version
            cleaned = clean_source(source)
            if not cleaned:
                cleaned = source # Fallback if cleaner is too aggressive for eval samples
            
            v = detect_pragma_version(cleaned)
            if not v:
                v = "0.4.24" # Safe fallback
            
            bucket = v[:3] # e.g. "0.4"
            
            record = {
                "contract_name": filename,
                "contract_address": filename, # This becomes contract_id in the shard
                "source_code": cleaned,
                "compiler_version": v
            }
            
            if bucket not in bucket_handles:
                bucket_handles[bucket] = open(EVAL_BUCKET_DIR / f"{bucket}.jsonl", 'w')
                bucket_counts[bucket] = 0
            
            bucket_handles[bucket].write(json.dumps(record) + '\n')
            bucket_counts[bucket] += 1
            
    for h in bucket_handles.values():
        h.close()
        
    print(f"Bucketed {sum(bucket_counts.values())} contracts into {len(bucket_counts)} files.")
    return bucket_counts

def run_extraction():
    # 1. Ensure solc versions are installed
    all_versions = set()
    for f in os.listdir(EVAL_BUCKET_DIR):
        if f.endswith(".jsonl"):
            with open(EVAL_BUCKET_DIR / f, 'r') as fh:
                for line in fh:
                    all_versions.add(json.loads(line)["compiler_version"])
    
    print(f"Pre-installing {len(all_versions)} solc versions...")
    preflight_install(list(all_versions))
    
    # 2. Run preprocess_contract.py (Node features)
    print("\n>>> Phase 1: Generating Node Features (preprocess_contract.py)")
    cmd_features = [
        PYTHON_BIN, str(DATA_PREP_DIR / "preprocess_contract.py"),
        "--batch", str(EVAL_BUCKET_DIR),
        "--output-dir", str(OUTPUT_DIR),
        "--workers", "16"
    ]
    import subprocess
    subprocess.run(cmd_features, check=True)
    
    # 3. Run compute_err.py (ERR pairs)
    print("\n>>> Phase 2: Generating ERR Pairs (compute_err.py)")
    for f in os.listdir(EVAL_BUCKET_DIR):
        if f.endswith(".jsonl"):
            bucket_name = Path(f).stem
            cmd_err = [
                PYTHON_BIN, str(DATA_PREP_DIR / "compute_err.py"),
                "--jsonl", str(EVAL_BUCKET_DIR / f),
                "--output-dir", str(OUTPUT_DIR),
                "--bucket-name", bucket_name,
                "--workers", "16"
            ]
            subprocess.run(cmd_err, check=True)

if __name__ == "__main__":
    prepare_eval_jsonl()
    run_extraction()
