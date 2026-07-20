import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import argparse
import sys
from tqdm import tqdm
import numpy as np

# Setup paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))

from evaluation.phase2.mil_loader import MILLoader, get_mil_splits
from models.graph_mil_head import GraphMILHead
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig


def train_graph_mil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # 2. Datasets
    label_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_labels.jsonl")
    ]
    annotation_files = [
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl"),
        os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/scrawld_annotations.jsonl")
    ]
    
    train_indices, val_indices, test_indices = get_mil_splits(label_files)
    
    if args.limit > 0:
        print(f"Limiting training data to {args.limit} samples.")
        import random
        random.seed(42)
        train_indices = random.sample(train_indices, min(len(train_indices), args.limit))
        val_indices = random.sample(val_indices, min(len(val_indices), args.limit // 5))
        test_indices = random.sample(test_indices, min(len(test_indices), args.limit // 5))

    full_dataset = MILLoader(
        label_files, annotation_files, tokenizer, 
        max_len=args.max_len, 
        npz_dir=args.npz_dir, 
        df_dir=args.df_dir
    )
    
    train_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices), 
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, val_indices), 
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(full_dataset, test_indices), 
        batch_size=1
    )
    
    # 3. Initialize Model
    if args.model_type == "wire":
        config = WIREConfig(d_head=64)
        encoder = WIREModernBERT(args.base_model_path, wire_config=config)
    else:
        encoder = AutoModel.from_pretrained(args.base_model_path)
        
    if args.checkpoint:
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Handle cases where encoder is wrapped or has different prefixes
        if any(k.startswith("encoder.") for k in sd.keys()):
             sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        elif any(k.startswith("backbone.") for k in sd.keys()):
             sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
             
        encoder.load_state_dict(sd, strict=False)
        
    if hasattr(encoder, "config"):
        hidden_dim = encoder.config.hidden_size
    else:
        hidden_dim = getattr(encoder, "hidden_size", 768)
    
    # Use the new Graph-Aware MIL Head
    mil_head = GraphMILHead(
        hidden_dim, 
        num_classes=8,
        num_gat_layers=args.num_gat_layers,
        num_attn_heads=args.num_attn_heads,
        graph_feature_dim=6,
        dropout=args.dropout,
    )
    
    encoder.to(device)
    mil_head.to(device)
    
    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr},
        {"params": mil_head.parameters(), "lr": args.lr * 5},  # MIL head learns faster
    ], weight_decay=0.01)
    
    # Cosine LR schedule with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader)  # 1 epoch warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    bag_criterion = nn.BCEWithLogitsLoss()
    instance_criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Load annotation quality info
    annotations = full_dataset.annotations
    curated_files = set()
    with open(os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("dataset") == "SBCurated":
                curated_files.add(d["filename"])
    
    print(f"High-quality (SBCurated) annotations: {len(curated_files)} files")
    print(f"Starting training: {args.epochs} epochs, lr={args.lr}")
    
    # 4. Training Loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        mil_head.train()
        train_loss = 0
        train_bag_loss = 0
        train_inst_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            
            # Forward pass through encoder
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                outputs = encoder(**kwargs)
            else:
                outputs = encoder(input_ids, attention_mask=attention_mask)
            
            # Extract last hidden state
            if isinstance(outputs, dict):
                token_embeddings = outputs.get("last_hidden_state")
                if token_embeddings is None:
                    token_embeddings = outputs.get("hidden_states")
                    if token_embeddings is None:
                        token_embeddings = next(iter(outputs.values()))
            else:
                token_embeddings = outputs[0]
            
            # Graph features for adjacency construction
            graph_feats = batch.get("graph_features")
            if graph_feats is not None:
                graph_feats = graph_feats.to(device)
            
            # Graph-Aware MIL Head
            mil_out = mil_head(
                token_embeddings,
                token_to_line=token_to_line,
                attention_mask=attention_mask,
                graph_features=graph_feats,
            )
            
            bag_logits = mil_out["bag_logits"]
            instance_logits = mil_out["instance_logits"]  # [B, L, num_classes]
            line_mask = mil_out["line_mask"]  # [B, L]
            
            # --- Bag-level loss ---
            loss_bag = bag_criterion(bag_logits, labels)
            
            # --- Instance-level auxiliary loss ---
            # For samples with annotations, apply weak instance supervision
            loss_inst = torch.tensor(0.0, device=device)
            inst_count = 0
            
            for b_idx in range(input_ids.size(0)):
                filename = batch["filename"][b_idx]
                if filename in annotations and filename in curated_files:
                    gt_lines = annotations[filename]
                    L = instance_logits.size(1)
                    
                    # Create instance-level labels: vulnerable lines get the bag label
                    inst_targets = torch.zeros(L, 8, device=device)
                    for line_idx in gt_lines:
                        if 0 < line_idx < L:
                            inst_targets[line_idx] = labels[b_idx]
                    
                    # Only compute loss on valid lines
                    valid_lines = line_mask[b_idx] > 0
                    if valid_lines.any():
                        inst_loss = instance_criterion(
                            instance_logits[b_idx][valid_lines],
                            inst_targets[valid_lines]
                        )
                        loss_inst = loss_inst + inst_loss.mean()
                        inst_count += 1
            
            if inst_count > 0:
                loss_inst = loss_inst / inst_count
            
            # Combined loss
            loss = loss_bag + args.instance_loss_weight * loss_inst
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(mil_head.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_bag_loss += loss_bag.item()
            if inst_count > 0:
                train_inst_loss += loss_inst.item()
            
            global_step += 1
        
        avg_loss = train_loss / len(train_loader)
        avg_bag = train_bag_loss / len(train_loader)
        avg_inst = train_inst_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} (bag={avg_bag:.4f} inst={avg_inst:.4f}) | LR: {lr_now:.2e}")
        
        # Validation
        if val_loader and (epoch + 1) % 2 == 0:
            val_loss = validate(encoder, mil_head, val_loader, device, args, bag_criterion)
            print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best checkpoint
                ckpt_dir = os.path.join(PROJECT_ROOT, f"Eval_Phase2/checkpoints/{args.model_name}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "encoder": encoder.state_dict(),
                    "mil_head": mil_head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, os.path.join(ckpt_dir, "best.pt"))
                print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")
    
    # Final Localization Evaluation
    evaluate_localization(encoder, mil_head, test_loader, device, args)


def validate(encoder, mil_head, val_loader, device, args, criterion):
    encoder.eval()
    mil_head.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                outputs = encoder(**kwargs)
            else:
                outputs = encoder(input_ids, attention_mask=attention_mask)
            
            if isinstance(outputs, dict):
                token_embeddings = outputs.get("last_hidden_state")
                if token_embeddings is None:
                    token_embeddings = outputs.get("hidden_states")
                    if token_embeddings is None:
                        token_embeddings = next(iter(outputs.values()))
            else:
                token_embeddings = outputs[0]
            
            graph_feats = batch.get("graph_features")
            if graph_feats is not None:
                graph_feats = graph_feats.to(device)
            
            mil_out = mil_head(
                token_embeddings,
                token_to_line=token_to_line,
                attention_mask=attention_mask,
                graph_features=graph_feats,
            )
            
            loss = criterion(mil_out["bag_logits"], labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def evaluate_localization(encoder, mil_head, test_loader, device, args):
    encoder.eval()
    mil_head.eval()
    
    # Separate tracking for SBCurated (reliable) vs ScrawlD (noisy)
    results_curated = {"top1": 0, "top3": 0, "top5": 0, "total": 0}
    results_scrawld = {"top1": 0, "top3": 0, "top5": 0, "total": 0}
    results_all = {"top1": 0, "top3": 0, "top5": 0, "total": 0}
    
    # Load dataset info for SBCurated identification
    curated_files = set()
    with open(os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/annotations.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("dataset") == "SBCurated":
                curated_files.add(d["filename"])
    
    output_vis_dir = os.path.join(PROJECT_ROOT, f"Eval_Phase2/localization_results/{args.model_name}")
    os.makedirs(output_vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Localization"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_to_line = batch["token_to_line"].to(device)
            filename = batch["filename"][0]
            
            annotations = test_loader.dataset.dataset.annotations
            if filename not in annotations:
                continue
            
            gt_lines = annotations[filename]
            is_curated = filename in curated_files
            
            if args.model_type == "wire":
                g_feats = batch.get("graph_features")
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if g_feats is not None:
                    kwargs["graph_features"] = g_feats.to(device)
                outputs = encoder(**kwargs)
            else:
                outputs = encoder(input_ids, attention_mask=attention_mask)
                
            if isinstance(outputs, dict):
                token_embeddings = outputs.get("last_hidden_state")
                if token_embeddings is None:
                    token_embeddings = outputs.get("hidden_states")
                    if token_embeddings is None:
                        token_embeddings = next(iter(outputs.values()))
            else:
                token_embeddings = outputs[0]
            
            graph_feats = batch.get("graph_features")
            if graph_feats is not None:
                graph_feats = graph_feats.to(device)
            
            # Graph-Aware MIL Head
            mil_out = mil_head(
                token_embeddings,
                token_to_line=token_to_line,
                attention_mask=attention_mask,
                graph_features=graph_feats,
            )
            
            # Line-level attention weights (already at line level, not token level!)
            attn_weights = mil_out["attention_weights"].squeeze(0).cpu().numpy()  # [L]
            line_mask = mil_out["line_mask"].squeeze(0).cpu().numpy()  # [L]
            
            # Also get instance logits for supplementary scoring
            inst_logits = mil_out["instance_logits"].squeeze(0).cpu().numpy()  # [L, 8]
            
            # Combined score: attention weight + max instance logit
            inst_max_score = inst_logits.max(axis=1) if inst_logits.shape[1] > 0 else np.zeros(len(attn_weights))
            combined_score = attn_weights + 0.3 * torch.sigmoid(torch.tensor(inst_max_score)).numpy()
            
            # Rank lines by combined score (only valid lines)
            line_scores = []
            for line_idx in range(len(combined_score)):
                if line_mask[line_idx] > 0 and line_idx > 0:
                    line_scores.append((line_idx, float(combined_score[line_idx])))
            
            sorted_lines = sorted(line_scores, key=lambda x: x[1], reverse=True)
            top_lines = [l for l, s in sorted_lines]
            
            # Update results
            for results in [results_all, results_curated if is_curated else results_scrawld]:
                results["total"] += 1
                if top_lines and top_lines[0] in gt_lines:
                    results["top1"] += 1
                if any(l in gt_lines for l in top_lines[:3]):
                    results["top3"] += 1
                if any(l in gt_lines for l in top_lines[:5]):
                    results["top5"] += 1
            
            # Save visualization for first 30 samples
            if results_all["total"] <= 30:
                save_vis(filename, sorted_lines, gt_lines, is_curated, output_vis_dir)
    
    # Print Results
    print(f"\n{'='*60}")
    print(f"  Localization Results: {args.model_name}")
    print(f"{'='*60}")
    
    for name, results in [("ALL", results_all), ("SBCurated (Human-Verified)", results_curated), ("ScrawlD (Auto-Generated)", results_scrawld)]:
        if results["total"] > 0:
            print(f"\n  [{name}] Total: {results['total']}")
            print(f"    Top-1 Accuracy: {results['top1']/results['total']:.4f}")
            print(f"    Top-3 Accuracy: {results['top3']/results['total']:.4f}")
            print(f"    Top-5 Accuracy: {results['top5']/results['total']:.4f}")
    
    # Save report
    report = {
        "model_name": args.model_name,
        "all": {k: v/max(1, results_all["total"]) if k != "total" else v for k, v in results_all.items()},
        "curated": {k: v/max(1, results_curated["total"]) if k != "total" else v for k, v in results_curated.items()},
        "scrawld": {k: v/max(1, results_scrawld["total"]) if k != "total" else v for k, v in results_scrawld.items()},
    }
    
    report_dir = os.path.join(PROJECT_ROOT, "Eval_Phase2/reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{args.model_name}_localization.json"), 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nReport saved to {report_dir}/{args.model_name}_localization.json")


def save_vis(filename, sorted_lines, gt_lines, is_curated, output_dir):
    safe_fn = filename.replace("/", "_")
    vis_path = os.path.join(output_dir, f"{safe_fn}_vis.txt")
    with open(vis_path, 'w') as f:
        f.write(f"Dataset: {'SBCurated (Human-Verified)' if is_curated else 'ScrawlD (Auto-Generated)'}\n")
        f.write(f"Vulnerable Lines (GT): {gt_lines}\n")
        f.write(f"Top Predicted Lines:\n")
        for i, (line, score) in enumerate(sorted_lines[:15]):
            status = "[HIT]" if line in gt_lines else ""
            f.write(f"  {i+1}. Line {line}: {score:.6f} {status}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["wire", "modernbert", "codebert", "graphcodebert"])
    parser.add_argument("--base_model_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--npz_dir", type=str, default=None)
    parser.add_argument("--df_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--num_gat_layers", type=int, default=2)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--instance_loss_weight", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size for fast benchmark")
    args = parser.parse_args()
    
    train_graph_mil(args)
