#!/usr/bin/env python3
"""
Benchmark 5: Progressive Typing Simulation (AST-Driven)
=======================================================
Evaluates "nuance awareness" and "early warning" at scale.
Uses `tree-sitter` to parse vulnerable/safe functions from the SBCurated dataset
and sequentially unmasks statements (simulating a developer typing line by line).

Measures the risk trajectory of Vanilla ModernBERT vs Exe-MLM v5 to see which
model detects the exact moment a vulnerability is typed, and which avoids false alarms
on safe variations (like Checks-Effects-Interactions).
"""

import os
import json
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tree_sitter import Language, Parser
import tree_sitter_solidity

import sys
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

LABELS_PATH = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry/labels.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/progressive_results_full")

# Anchors
REENTRANCY_ANCHOR = """
pragma solidity ^0.6.0;
contract Hack {
    mapping(address => uint) balances;
    function attack() external {
        payable(msg.sender).call{value: balances[msg.sender]}("");
        balances[msg.sender] = 0;
    }
}
"""

SAFE_ANCHOR = """
pragma solidity ^0.8.0;
contract Safe {
    mapping(address => uint) balances;
    function withdraw(uint amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
    }
}
"""

def setup_parser():
    SOLIDITY_LANGUAGE = Language(tree_sitter_solidity.language())
    parser = Parser()
    parser.language = SOLIDITY_LANGUAGE
    return parser

def find_enclosing_function(node, start_byte, end_byte):
    if node.start_byte <= start_byte and node.end_byte >= end_byte:
        if node.type == 'function_definition':
            return node
        for child in node.children:
            res = find_enclosing_function(child, start_byte, end_byte)
            if res:
                return res
    return None

def get_emb(model, tokenizer, text, device, is_wire=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding="max_length").to(device)
    with torch.no_grad():
        if is_wire:
            g = torch.zeros((1, 512, 6), device=device)
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"], graph_features=g)
        else:
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        hidden = out.get("hidden_states", out.get("last_hidden_state")) if isinstance(out, dict) else out.last_hidden_state
        return torch.mean(hidden, dim=1).cpu().numpy().squeeze()

def simulate_typing(source_bytes, func_node):
    # Extract signature and body
    body_node = None
    for child in func_node.children:
        if child.type in ['function_body', 'block']:
            body_node = child
            break
            
    if not body_node:
        return []

    # Get signature text (everything before the block)
    sig_text = source_bytes[func_node.start_byte:body_node.start_byte].decode('utf8', errors='ignore')
    
    stages = [sig_text + "{}"] # Stage 0: empty body
    
    # Extract statements inside the body block
    statements = []
    for child in body_node.children:
        if child.type not in ['{', '}']:
            statements.append(source_bytes[child.start_byte:child.end_byte].decode('utf8', errors='ignore'))
            
    current_body = ""
    for stmt in statements:
        current_body += "    " + stmt + "\n"
        stages.append(sig_text + "{\n" + current_body + "}")
        
    return stages

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    parser = setup_parser()
    reentrancy_samples = []
    print("Loading Full Dataset (Reentrancy & Safe) contracts...")
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            labels = rec.get("labels", {})
            if labels.get("Reentrancy") == 1:
                reentrancy_samples.append((rec, True))
            # Grab some clean samples as well
            elif sum(labels.values()) == 0:
                reentrancy_samples.append((rec, False))
                    
    import random
    random.seed(42)
    v_samples = [s for s in reentrancy_samples if s[1]]
    s_samples = [s for s in reentrancy_samples if not s[1]]
    
    # Balance the dataset (take min count of either safe or vuln)
    min_count = min(len(v_samples), len(s_samples))
    v_samples = random.sample(v_samples, min_count)
    s_samples = random.sample(s_samples, min_count)
    eval_samples = v_samples + s_samples
    random.shuffle(eval_samples)
    print(f"Loaded {len(v_samples)} vulnerable and {len(s_samples)} safe samples (Total: {len(eval_samples)}).")

    # 2. Setup Models
    base = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base)
    
    print("Loading Vanilla ModernBERT...")
    modernbert = AutoModel.from_pretrained(base).to(device)
    modernbert.eval()
    
    print("Loading Exe-MLM v5...")
    config = WIREConfig(d_head=64, n_features=6)
    exe_mlm = WIREModernBERT(base, wire_config=config).to(device)
    
    # Disable reference compile for trace/eval compat
    exe_mlm.bert_config.reference_compile = False
    exe_mlm.encoder.config.reference_compile = False
    if hasattr(exe_mlm.encoder, "compiled_embeddings"):
        exe_mlm.encoder.compiled_embeddings = exe_mlm.encoder.embeddings
        
    ckpt_path = os.path.join(PROJECT_ROOT, "Eval_Phase2/checkpoints/exe_mlm_v5_llvt_text_only/best.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        exe_mlm.load_state_dict(sd, strict=False)
    exe_mlm.eval()

    # Get anchors
    mb_anchor_r = get_emb(modernbert, tokenizer, REENTRANCY_ANCHOR, device, False)
    mb_anchor_s = get_emb(modernbert, tokenizer, SAFE_ANCHOR, device, False)
    exe_anchor_r = get_emb(exe_mlm, tokenizer, REENTRANCY_ANCHOR, device, True)
    exe_anchor_s = get_emb(exe_mlm, tokenizer, SAFE_ANCHOR, device, True)

    # 3. Progressive Typing Simulation
    mb_trajectories = {"vuln": [], "safe": []}
    exe_trajectories = {"vuln": [], "safe": []}
    
    print(f"Starting rigorous benchmark on {len(eval_samples)} contracts...")
    for idx, (rec, is_vuln) in enumerate(eval_samples):
        src = rec["source_code"]
        src_bytes = bytes(src, "utf8")
        tree = parser.parse(src_bytes)
        root = tree.root_node
        
        # Find function to simulate
        func_node = None
        import re
        match = re.search(r'@vulnerable_at_lines:\s*([0-9,]+)', src)
        if match and is_vuln:
            vuln_line = int(match.group(1).split(',')[0]) - 1
            lines = src.split("\\n")
            if vuln_line < len(lines):
                target_start = sum(len(l) + 1 for l in lines[:vuln_line])
                target_end = target_start + len(lines[vuln_line])
                func_node = find_enclosing_function(root, target_start, target_end)
        
        if not func_node:
            def get_first_func(n):
                if n.type == 'function_definition':
                    return n
                for c in n.children:
                    res = get_first_func(c)
                    if res: return res
                return None
            func_node = get_first_func(root)
            
        if not func_node:
            continue
            
        stages = simulate_typing(src_bytes, func_node)
        if len(stages) < 3: continue # skip trivial functions
        
        # Evaluate stages
        mb_traj, exe_traj = [], []
        for stage_code in stages:
            # Wrap in dummy contract to ensure valid parseable syntax for embedding
            stage_wrapped = f"pragma solidity ^0.8.0;\\ncontract C {{\\n{stage_code}\\n}}"
            
            mb_emb = get_emb(modernbert, tokenizer, stage_wrapped, device, False)
            exe_emb = get_emb(exe_mlm, tokenizer, stage_wrapped, device, True)
            
            # Risk score: dist(safe) - dist(risky). Positive = closer to risky.
            mb_risk = cosine(mb_emb, mb_anchor_s) - cosine(mb_emb, mb_anchor_r)
            exe_risk = cosine(exe_emb, exe_anchor_s) - cosine(exe_emb, exe_anchor_r)
            
            mb_traj.append(float(mb_risk))
            exe_traj.append(float(exe_risk))
            
        # Normalize to % of function typed (0.0 to 1.0)
        x_pct = np.linspace(0, 1, len(stages)).tolist()
        label = "vuln" if is_vuln else "safe"
        
        mb_trajectories[label].append((x_pct, mb_traj))
        exe_trajectories[label].append((x_pct, exe_traj))
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(eval_samples):
            print(f"Processed {idx+1}/{len(eval_samples)} contracts...")

    # 4. Plot Trajectories
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    def plot_trajectories(ax, trajs, title):
        # Interpolate onto common x grid
        common_x = np.linspace(0, 1, 50)
        
        v_interp = []
        for x, y in trajs["vuln"]:
            if len(x) > 1:
                v_interp.append(np.interp(common_x, x, y))
                
        s_interp = []
        for x, y in trajs["safe"]:
            if len(x) > 1:
                s_interp.append(np.interp(common_x, x, y))
                
        if v_interp:
            v_mean = np.mean(v_interp, axis=0)
            v_std = np.std(v_interp, axis=0)
            ax.plot(common_x, v_mean, color="red", label="Vulnerable Contracts", lw=2)
            ax.fill_between(common_x, v_mean - v_std, v_mean + v_std, color="red", alpha=0.1)
            
        if s_interp:
            s_mean = np.mean(s_interp, axis=0)
            s_std = np.std(s_interp, axis=0)
            ax.plot(common_x, s_mean, color="green", label="Safe Contracts", lw=2)
            ax.fill_between(common_x, s_mean - s_std, s_mean + s_std, color="green", alpha=0.1)
            
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Progressive Typing (% of Function Complete)")
        ax.set_ylabel("Risk Score (Closeness to Reentrancy Anchor)")
        ax.axhline(0, color='gray', linestyle='--')
        ax.legend()

    plot_trajectories(axes[0], mb_trajectories, "Vanilla ModernBERT")
    plot_trajectories(axes[1], exe_trajectories, "Exe-MLM v5 (Trajectory-Aware)")
    
    fig.suptitle("Progressive Typing Simulation: Risk Trajectories over Time", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "progressive_trajectories.png"))
    
    # Save data
    out_data = {"modernbert": mb_trajectories, "exe_mlm": exe_trajectories}
    with open(os.path.join(OUT_DIR, "progressive_data.json"), "w") as f:
        json.dump(out_data, f)
        
    print(f"\\nDone! Saved plots and data to {OUT_DIR}")

if __name__ == "__main__":
    main()
