# EAGLE: Execution-Aware Graph Learning for Smart Contract Vulnerability Detection

> **Anonymous submission** — code provided for reproducibility during review.

## Overview

EAGLE is a framework for smart contract vulnerability detection that combines:

- **Graph-RoPE**: Graph-aware Rotary Position Embeddings that inject Code Property Graph (CPG) structural information into transformer attention via effective resistance distance.
- **Execution-aware Pre-training**: Masked language model pre-training with execution trajectory awareness for understanding smart contract semantics.
- **Multi-Instance Learning (MIL)**: Contract-level vulnerability classification with fine-grained line-level localization.

## Repository Structure

```
eagle/
├── configs/                # Training & experiment configurations
│   ├── baseline/           # Graph-RoPE baseline configs
│   ├── wire/               # WIRE ablation configs
│   ├── metp/               # METP configs
│   └── vanilla/            # Vanilla MLM configs
├── data_prep/              # CPG extraction, ERR computation, dataset building
├── models/                 # Model architectures
│   ├── graph_rope.py       # Core Graph-RoPE implementation
│   ├── graph_rope_model.py # Full Graph-RoPE model
│   ├── llvt_model.py       # LLVT (Line-Level Vulnerability Tagger)
│   ├── graph_mil_head.py   # Graph-aware MIL head
│   ├── baseline/           # Baseline model variants
│   ├── wire/               # WIRE model variant
│   ├── metp/               # METP model variant
│   ├── exe_graph_mlm/      # Execution-aware graph MLM
│   └── shared/             # Shared datasets & utilities
├── experiments/            # Training scripts for all model variants
├── evaluation/             # Full evaluation pipeline
│   ├── phase1/             # Tier-1 detection evaluation
│   ├── phase2/             # Tier-2 localization evaluation
│   ├── benchmarks/         # Structured benchmarks (Tier-1/2/3)
│   ├── semantic/           # Semantic robustness benchmarks
│   └── scripts/            # Data ingestion & preprocessing
├── eagle_engine/           # LSP-based real-time inference server
├── eagle_engine_export/    # ONNX model export pipeline
├── diff_based_detection/   # Diff-based vulnerability detection module
├── research/               # Ablation studies
├── tests/                  # Unit tests
└── utils/                  # Data analysis utilities
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducibility

### 1. Data Preparation

```bash
# Extract CPG features from Solidity contracts
python data_prep/preprocess_contract.py --input <contracts_dir> --output <features_dir>

# Compute Effective Resistance (ERR) features
python data_prep/compute_err.py --feat-dir <features_dir>
```

### 2. Pre-training

```bash
# EAGLE (Graph-RoPE + Exe-MLM) — main model
python experiments/train_graph_rope.py --config configs/baseline/config_e1.json

# Baseline (vanilla MLM)
python experiments/vanilla/train_vanilla_mlm.py --config configs/vanilla/config_vanilla_mlm.json

# WIRE ablation
python experiments/wire/train_wire.py --config configs/wire/config_wire.json
```

### 3. Evaluation

```bash
# Run Tier-1 detection benchmarks
python evaluation/master_benchmark.py

# Run semantic robustness benchmarks
python evaluation/semantic/orchestrator_semantic.py
```

### 4. Inference Engine

```bash
# Start the EAGLE LSP server for real-time analysis
python eagle_engine/main.py
```

## Configuration

All experiments are configured via JSON files in `configs/`. Set the `PROJECT_ROOT` environment variable to point to your data directory:

```bash
export PROJECT_ROOT=/path/to/your/data
```

## Requirements

See `requirements.txt` for the full dependency list. Key dependencies:
- PyTorch ≥ 2.0
- HuggingFace Transformers ≥ 4.30
- Slither (Solidity static analyzer)
- NetworkX (graph operations)
