import argparse
import sys
import logging
from transformers import AutoTokenizer
import os
sys.path.insert(0, os.path.dirname(__file__))
from inference import EagleEngine
from server import create_server
from tokenizer_utils import prepare_inputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_anchors(engine, tokenizer, anchors_path):
    """
    Runs the 20 canonical anchor snippets through the model on startup 
    to extract their 64D manifold centroids. This ensures the anchors 
    are perfectly synchronized with the model weights (even if a LoRA is loaded).
    """
    logger.info("Pre-computing dynamic anchors...")
    
    vuln_embs = []
    safe_embs = []
    
    for anchor in engine.raw_anchors:
        chunks, token_to_line, total = prepare_inputs(anchor["code"], tokenizer)
        # Extract the pure vulnerable/safe manifold representation by taking the embedding
        # of the line that the model considers most vulnerable in the snippet.
        line_scores, line_embs, _ = engine.run_document(chunks)
        if len(line_embs) == 0:
            continue
            
        import numpy as np
        max_line = max(line_scores.keys(), key=lambda l: line_scores[l])
        centroid = line_embs[max_line]
        
        if anchor["is_safe"]:
            safe_embs.append(centroid)
        else:
            vuln_embs.append(centroid)
            
    engine.set_anchors(vuln_embs, safe_embs)
    logger.info(f"Successfully generated {len(vuln_embs)} vulnerable and {len(safe_embs)} safe anchors.")

def main():
    parser = argparse.ArgumentParser(description="EAGLE LSP Server")
    parser.add_argument("--onnx", type=str, required=True, help="Path to eagle.onnx")
    parser.add_argument("--anchors", type=str, required=True, help="Path to anchors.jsonl")
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base", help="HF Tokenizer name")
    
    args = parser.parse_args()
    
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info("Initializing ONNX Engine...")
    engine = EagleEngine(args.onnx, args.anchors)
    
    precompute_anchors(engine, tokenizer, args.anchors)
    
    logger.info("Starting EAGLE Language Server on STDIO...")
    server = create_server(engine, tokenizer)
    server.start_io()

if __name__ == "__main__":
    main()
