# EAGLE: Execution-Aware Graph Learning Encoder for Smart Contract Vulnerability Detection

> **Anonymous submission** — code provided for reproducibility during review.

## Overview

EAGLE is a structure-aware large language model designed for joint smart contract vulnerability detection and localization. It models programs through their execution geometry rather than linear token sequences. Key components include:

- **Code Property Graph (CPG) & Structural Coordinates**: Deriving continuous coordinates from Poisson flow potentials and multi-scale diffusion to represent execution structure.
- **Topology-Aware Positional Encoding (Graph-RoPE)**: Injecting the CPG structural information directly into transformer attention, replacing traditional sequential positions.
- **Geometry-Guided Self-Supervised Pre-Training**: A multi-objective framework including:
  - **Node-MLM**: Node-Level Masked Language Modeling
  - **METP**: Masked Execution Trajectory Prediction (Trajectory-Aware pre-training)
  - **TFR & ERR**: Topological Feature Regression and Effective Resistance Regression
- **Line-Level Vulnerability Tagger (LLVT)**: A sequence-labeling formulation for fine-grained, line-level vulnerability localization.

## Repository Structure & Paper Mapping

*Note: Some directory names reflect internal development code names (e.g., `wire` or `exe_graph_mlm`) which correspond to specific ablation models in the paper.*

```
eagle/
├── configs/                # Training & experiment configurations
│   ├── baseline/           # Main EAGLE (Trajectory-Aware) configs
│   ├── wire/               # Ablation configs
│   ├── metp/               # METP specific configs
│   └── vanilla/            # Vanilla ModernBERT configs
├── data_prep/              # CPG extraction, solver for Poisson resistance, dataset building
├── models/                 # Model architectures
│   ├── graph_rope.py       # Core Topology-Aware Positional Encoding (Graph-RoPE)
│   ├── graph_rope_model.py # Full EAGLE model architecture
│   ├── llvt_model.py       # LLVT (Line-Level Vulnerability Tagger) architecture
│   ├── exe_graph_mlm/      # Execution-aware graph MLM (Node-MLM/METP objectives)
│   └── baseline/           # Baseline and ablation variants
├── experiments/            # Training scripts for all model variants
├── evaluation/             # Full evaluation pipeline
│   ├── phase1/             # Tier-1 Vulnerability Detection (Contract-level)
│   ├── phase2/             # Tier-2 Vulnerability Localization (Line-level LLVT)
│   ├── benchmarks/         # Structured benchmarks (SBCurated, ScrawlD)
│   ├── semantic/           # Semantic robustness & Adversarial obfuscation benchmarks
│   └── scripts/            # Data ingestion & preprocessing
├── eagle_engine/           # LSP-based real-time inference server (IDE integration)
├── eagle_engine_export/    # ONNX model export for real-time JIT evaluation
├── diff_based_detection/   # Diff-based progressive typing simulation
├── research/               # Additional ablation studies
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

### 1. Data Preparation (Graph Compilation)

```bash
# Extract CPG features from Solidity contracts
python data_prep/preprocess_contract.py --input <contracts_dir> --output <features_dir>

# Compute Structural Coordinates (Poisson Resistance & Entropy)
python data_prep/compute_err.py --feat-dir <features_dir>
```

### 2. Pre-training & Fine-Tuning

```bash
# Train EAGLE (Trajectory-Aware) — Main Model
python experiments/train_graph_rope.py --config configs/baseline/config_e1.json

# Train Vanilla ModernBERT Baseline
python experiments/vanilla/train_vanilla_mlm.py --config configs/vanilla/config_vanilla_mlm.json

# Train Ablation Models
python experiments/wire/train_wire.py --config configs/wire/config_wire.json
```

### 3. Evaluation

```bash
# Run Vulnerability Detection and Localization Benchmarks (SBCurated)
python evaluation/master_benchmark.py

# Run Semantic Robustness and Adversarial Obfuscation Benchmarks
python evaluation/semantic/orchestrator_semantic.py
```

### 4. Real-Time IDE Integration (Inference Engine)

```bash
# Start the EAGLE LSP server for real-time progressive typing analysis
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
