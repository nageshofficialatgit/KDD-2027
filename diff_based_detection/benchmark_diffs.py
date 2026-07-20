import os
import json
import time
import torch
import argparse
import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Implementation"))
from models.wire.wire_rope_model import WIREModernBERT
from models.wire.wire_rope import WIREConfig

DATA_PATH = "./diff_based_detection/data/SBCurated_Diffs.jsonl"
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base" 
MAX_LEN_FULL = 2048
MAX_LEN_DIFF = 512

def measure_latency(model, input_ids, attention_mask, graph_features=None, num_runs=5):
    # Warmup
    for _ in range(2):
        if graph_features is not None:
            _ = model(input_ids, attention_mask, graph_features)
        else:
            _ = model(input_ids, attention_mask)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_runs):
        if graph_features is not None:
            _ = model(input_ids, attention_mask, graph_features)
        else:
            _ = model(input_ids, attention_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="modernbert", choices=["modernbert", "exe_mlm"])
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    print(f"Loading {args.model_type} for benchmarking...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    
    if args.model_type == "exe_mlm":
        model = WIREModernBERT(DEFAULT_MODEL_NAME, wire_config=WIREConfig(d_head=64, n_features=6))
        
        # Disable internal compilation for JIT tracing
        model.bert_config.reference_compile = False
        model.encoder.config.reference_compile = False
        if hasattr(model.encoder, "compiled_embeddings"):
            model.encoder.compiled_embeddings = model.encoder.embeddings
            
        if args.checkpoint:
            print(f"Loading weights from {args.checkpoint}...")
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            if any(k.startswith("encoder.") for k in sd.keys()):
                 sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
            elif any(k.startswith("backbone.") for k in sd.keys()):
                 sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
            model.load_state_dict(sd, strict=False)
        model = model.to(device)
    else:
        config = AutoConfig.from_pretrained(DEFAULT_MODEL_NAME)
        config.reference_compile = False # Disable internal torch.compile which blocks JIT
        config.return_dict = False # For JIT tracing
        model = AutoModel.from_pretrained(DEFAULT_MODEL_NAME, config=config).to(device)
        
    model.eval()
    
    print("JIT tracing models for both architectures (Full File = 2048, Diff = 512)...")
    dummy_input_full = torch.zeros(1, 2048, dtype=torch.long).to(device)
    dummy_mask_full = torch.ones(1, 2048, dtype=torch.long).to(device)
    
    dummy_input_diff = torch.zeros(1, 512, dtype=torch.long).to(device)
    dummy_mask_diff = torch.ones(1, 512, dtype=torch.long).to(device)
    
    if args.model_type == "exe_mlm":
        dummy_graph_full = torch.zeros(1, 2048, 6, dtype=torch.float32).to(device)
        dummy_graph_diff = torch.zeros(1, 512, 6, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            traced_model_full = torch.jit.trace(model, (dummy_input_full, dummy_mask_full, dummy_graph_full), strict=False)
            traced_model_diff = torch.jit.trace(model, (dummy_input_diff, dummy_mask_diff, dummy_graph_diff), strict=False)
    else:
        # Trace the models for performance optimization
        with torch.no_grad():
            traced_model_full = torch.jit.trace(model, (dummy_input_full, dummy_mask_full), strict=False)
            traced_model_diff = torch.jit.trace(model, (dummy_input_diff, dummy_mask_diff), strict=False)
    
    print(f"Loading diff dataset from {DATA_PATH}...")
    full_file_latency = []
    diff_latency = []
    
    count = 0
    with open(DATA_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            source_code = record["source_code"]
            diff_chunk = record.get("diff_chunk", "")
            
            if not diff_chunk:
                continue
                
            # Full file inference (2048 tokens)
            tokens_full = tokenizer(source_code, max_length=2048, padding="max_length", truncation=True, return_tensors="pt")
            ids_full = tokens_full["input_ids"].to(device)
            mask_full = tokens_full["attention_mask"].to(device)
            
            # Diff chunk inference (512 tokens)
            tokens_diff = tokenizer(diff_chunk, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            ids_diff = tokens_diff["input_ids"].to(device)
            mask_diff = tokens_diff["attention_mask"].to(device)
            
            with torch.no_grad():
                if args.model_type == "exe_mlm":
                    g_full = torch.zeros(1, 2048, 6, dtype=torch.float32).to(device)
                    g_diff = torch.zeros(1, 512, 6, dtype=torch.float32).to(device)
                    ms_full = measure_latency(traced_model_full, ids_full, mask_full, g_full, num_runs=5)
                    ms_diff = measure_latency(traced_model_diff, ids_diff, mask_diff, g_diff, num_runs=5)
                else:
                    ms_full = measure_latency(traced_model_full, ids_full, mask_full, num_runs=5)
                    ms_diff = measure_latency(traced_model_diff, ids_diff, mask_diff, num_runs=5)
                    
                full_file_latency.append(ms_full)
                diff_latency.append(ms_diff)
            
            count += 1
            if count % 10 == 0:
                print(f"Benchmarked {count} files...")
            if count >= 30: # Benchmark on 30 samples to avoid long waits
                break
                
    avg_full = sum(full_file_latency) / len(full_file_latency)
    avg_diff = sum(diff_latency) / len(diff_latency)
    
    print("\n" + "="*50)
    print("🎯 JIT-Compiled Inference Benchmarks (ModernBERT)")
    print("="*50)
    print(f"Samples Evaluated: {count}")
    print(f"Full-File Average Latency:  {avg_full:.2f} ms")
    print(f"Diff-Chunk Average Latency: {avg_diff:.2f} ms")
    print(f"Speedup Factor:             {avg_full/avg_diff:.2f}x")
    print("="*50)
    
if __name__ == "__main__":
    main()
