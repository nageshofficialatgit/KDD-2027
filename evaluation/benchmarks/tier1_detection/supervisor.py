import os
import optuna
import logging
import sys
from datetime import datetime
from benchmark_supervised import train_eval

# Ensure unified logging
LOG_DIR = "./evaluation/results/tier1_detection/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def objective(trial, model_type, mode, checkpoint, limit):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    
    trial_name = f"trial_{trial.number}"
    log_path = os.path.join(LOG_DIR, f"{model_type}_{mode}_{trial_name}.log")
    
    # Setup trial-specific logging
    # Note: benchmark_supervised uses print, we might want to redirect sys.stdout
    # For now, we'll rely on the return value for optimization
    
    args = [
        "--model_type", model_type,
        "--mode", mode,
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--epochs", "3", # Fewer epochs for tuning
        "--batch_size", "8",
        "--output_suffix", trial_name
    ]
    
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    if limit:
        args.extend(["--limit", str(limit)])
        
    try:
        val_f1 = train_eval(args)
        return val_f1
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

def run_experiment(model_type, mode, checkpoint=None, n_trials=5, limit=0):
    study_name = f"{model_type}_{mode}_study"
    print(f"\n>>> Starting HPO Study: {study_name}")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_type, mode, checkpoint, limit), n_trials=n_trials)
    
    print(f"Best Trial for {study_name}:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
    # Run final appraisal with best params on full evaluation (or more epochs)
    print(f">>> Running final evaluation for {model_type}_{mode} with best params...")
    best_args = [
        "--model_type", model_type,
        "--mode", mode,
        "--lr", str(study.best_params["lr"]),
        "--weight_decay", str(study.best_params["weight_decay"]),
        "--epochs", "5",
        "--batch_size", "8",
        "--output_suffix", "best_hpo"
    ]
    if checkpoint:
        best_args.extend(["--checkpoint", checkpoint])
    if limit:
        best_args.extend(["--limit", str(limit)])
        
    train_eval(best_args)

if __name__ == "__main__":
    # Execution Order
    WIRE_CHECKPOINT = "./checkpoints/e1_err/best.pt"
    
    # 1. WIRE Linear Probe (Done)
    # run_experiment("wire", "probe", checkpoint=WIRE_CHECKPOINT, n_trials=3, limit=1000)
    
    # 2. ModernBERT Baseline (Done)
    # run_experiment("modernbert", "finetune", n_trials=3, limit=1000)
    
    # 3. CodeBERT Baseline (Resume)
    run_experiment("codebert", "finetune", n_trials=3, limit=1000)
    
    # 4. WIRE Full Fine-tuning
    run_experiment("wire", "finetune", checkpoint=WIRE_CHECKPOINT, n_trials=3, limit=1000)
