#!/bin/bash
# Phase 1: Tier 1 Supervised Benchmark Suite

BASE_DIR="./evaluation/benchmarks/tier1_detection"
CKPT="./checkpoints/e1_err/best.pt"

echo "=== Starting Full Supervised Benchmark Suite ==="

# 1. WIRE Models
echo "Running WIRE (Linear Probe)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type wire --mode probe --checkpoint $CKPT --epochs 5 --batch_size 16

echo "Running WIRE (Fine-Tuning)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type wire --mode finetune --checkpoint $CKPT --epochs 3 --batch_size 8 --lr 2e-5

# 2. ModernBERT Baselines
echo "Running ModernBERT (Linear Probe)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type modernbert --mode probe --epochs 5 --batch_size 16

echo "Running ModernBERT (Fine-Tuning)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type modernbert --mode finetune --epochs 3 --batch_size 8 --lr 2e-5

# 3. CodeBERT Baselines
echo "Running CodeBERT (Linear Probe)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type codebert --mode probe --epochs 5 --batch_size 16

echo "Running CodeBERT (Fine-Tuning)..."
python3 $BASE_DIR/benchmark_supervised.py --model_type codebert --mode finetune --epochs 3 --batch_size 8 --lr 2e-5

echo "=== Benchmark Suite Complete ==="
