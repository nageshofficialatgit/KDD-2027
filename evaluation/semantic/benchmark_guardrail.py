#!/usr/bin/env python3
"""
Benchmark 4: Real-Time Guardrail Sensitivity — Early Warning
============================================================
Simulates a developer writing a vulnerable function incrementally in 4 stages.
Measures at which stage each model first "senses" the danger — before it is complete.

Stage 1: Function signature only (no risk)
Stage 2: State read (balance read — still safe)
Stage 3: External call BEFORE state update (THE dangerous moment)
Stage 4: State update after external call (Reentrancy complete)

Method: Track cosine distance to a "Reentrancy" anchor at each stage.
Metric: "Alert Stage" — the earliest stage where distance drops below threshold.
Lower stage = smarter, earlier warning.
"""
import torch
import json
import os
import sys
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT, WIREConfig

# The four incremental stages of writing a Reentrancy vulnerability
STAGES = {
    1: """
pragma solidity ^0.8.0;
contract Wallet {
    mapping(address => uint) public balances;
    
    function withdraw(uint amount) public {
        // TODO: implement
    }
}
""",
    2: """
pragma solidity ^0.8.0;
contract Wallet {
    mapping(address => uint) public balances;
    
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient");
        uint toSend = amount;
        // TODO: send and update
    }
}
""",
    3: """
pragma solidity ^0.8.0;
contract Wallet {
    mapping(address => uint) public balances;
    
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient");
        uint toSend = amount;
        (bool success, ) = msg.sender.call{value: toSend}("");
        require(success, "Transfer failed");
        // TODO: update state
    }
}
""",
    4: """
pragma solidity ^0.8.0;
contract Wallet {
    mapping(address => uint) public balances;
    
    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient");
        uint toSend = amount;
        (bool success, ) = msg.sender.call{value: toSend}("");
        require(success, "Transfer failed");
        balances[msg.sender] -= toSend;  // STATE UPDATE AFTER EXTERNAL CALL = REENTRANCY
    }
}
"""
}

# Reentrancy anchor — a known bad pattern
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

# Safe anchor — checks-effects-interactions pattern
SAFE_ANCHOR = """
pragma solidity ^0.8.0;
contract Safe {
    mapping(address => uint) balances;
    function withdraw(uint amount) external {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;  // State update FIRST
        payable(msg.sender).transfer(amount);  // External call AFTER
    }
}
"""

ALERT_THRESHOLD = 0.05  # Alert if within this cosine distance of the Reentrancy anchor

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

def evaluate_guardrail(model, tokenizer, device, is_wire):
    anchor_risky = get_emb(model, tokenizer, REENTRANCY_ANCHOR, device, is_wire)
    anchor_safe  = get_emb(model, tokenizer, SAFE_ANCHOR,       device, is_wire)

    stage_results = {}
    alert_stage = None

    for stage_num, code in STAGES.items():
        emb = get_emb(model, tokenizer, code, device, is_wire)
        dist_risky = cosine(emb, anchor_risky)
        dist_safe  = cosine(emb, anchor_safe)
        risk_score = dist_safe - dist_risky   # Positive = leaning toward risky anchor
        alerted    = dist_risky < ALERT_THRESHOLD

        stage_results[stage_num] = {
            "dist_to_risky_anchor": float(dist_risky),
            "dist_to_safe_anchor": float(dist_safe),
            "risk_score": float(risk_score),
            "alerted": alerted,
        }

        if alerted and alert_stage is None:
            alert_stage = stage_num

    return {
        "stages": stage_results,
        "first_alert_stage": alert_stage if alert_stage else "Never",
        "detected_before_complete": alert_stage is not None and alert_stage < 4,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base)

    results = {}

    # --- Vanilla ModernBERT ---
    print("[1/2] Vanilla ModernBERT — Guardrail Trace")
    vanilla = AutoModel.from_pretrained(base).to(device)
    vanilla.eval()
    results["ModernBERT (Vanilla)"] = evaluate_guardrail(vanilla, tokenizer, device, False)
    del vanilla; torch.cuda.empty_cache()

    # --- Exe-MLM v5 ---
    print("[2/2] Exe-MLM v5 — Guardrail Trace")
    config = WIREConfig(d_head=64)
    wire = WIREModernBERT(base, wire_config=config).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/cache/exe_graph_mlm_checkpoints/exe_mlm_best.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        wire.load_state_dict(sd, strict=False)
    wire.eval()
    results["Exe-MLM v5"] = evaluate_guardrail(wire, tokenizer, device, True)
    del wire; torch.cuda.empty_cache()

    # --- Report ---
    print("\n" + "=" * 65)
    print("BENCHMARK 4: REAL-TIME GUARDRAIL — Early Warning by Stage")
    print("=" * 65)

    stage_labels = {
        1: "Signature Only",
        2: "State Read",
        3: "External Call (Dangerous!)",
        4: "State Update (Reentrancy Complete)",
    }

    for model_name, res in results.items():
        print(f"\n{model_name}")
        print(f"  First Alert Stage: {res['first_alert_stage']}")
        print(f"  Early Warning (before Stage 4): {'✅ YES' if res['detected_before_complete'] else '❌ NO'}")
        print(f"  {'Stage':<5} {'Description':<32} {'Risk Score':>11} {'Alerted':>8}")
        print(f"  {'-'*60}")
        for s, sr in res["stages"].items():
            alerted_str = "⚠️  YES" if sr["alerted"] else "      no"
            print(f"  {s:<5} {stage_labels[s]:<32} {sr['risk_score']:>+11.5f} {alerted_str:>8}")

    print("\n" + "=" * 65)

    out_path = os.path.join(PROJECT_ROOT,
        "Implementation/evaluation/semantic/b4_guardrail_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
