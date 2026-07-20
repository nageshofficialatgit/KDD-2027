import os
import json
import sys
from pathlib import Path
import numpy as np

# Setup paths
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(PROJECT_ROOT / "Implementation"))

from data_prep.graph_builder import build_cpg_from_source

def extract_dataflow(source_code):
    """
    Extracts GraphCodeBERT-style data flow from Solidity source.
    Returns:
        variable_sequence: List of (var_name, [line_numbers])
        dfg_edges: List of (src_idx, dst_idx) representing use-def chains in variable_sequence
    """
    cpg = build_cpg_from_source(source_code)
    if not cpg:
        return [], []
        
    nodes = cpg.nodes
    A_dfg = cpg.A_dfg.tocoo()
    
    # 1. Map IR nodes to variable occurrences
    # Each node in Slither can read/write multiple variables.
    variable_sequence = []
    node_to_var_indices = {} # node_id -> list of indices in variable_sequence
    
    for node in nodes:
        node_vars = []
        # Combine all mentioned variables in this node
        # (Slither doesn't give us exact token offsets within the node's expression easily,
        # but we know which lines they are on)
        vars_in_node = list(set(node.variables_read + node.variables_written))
        
        for vname in vars_in_node:
            idx = len(variable_sequence)
            variable_sequence.append({
                "name": vname,
                "lines": node.source_lines
            })
            node_vars.append(idx)
        
        node_to_var_indices[node.id] = node_vars
        
    # 2. Extract DFG edges
    dfg_edges = []
    # A_dfg has edges def(v)@i -> use(v)@j
    for i, j, w in zip(A_dfg.row, A_dfg.col, A_dfg.data):
        # i is source node, j is target node
        src_vars = node_to_var_indices.get(i, [])
        dst_vars = node_to_var_indices.get(j, [])
        
        # Connect variables with same name
        for sv_idx in src_vars:
            sv_name = variable_sequence[sv_idx]["name"]
            for dv_idx in dst_vars:
                dv_name = variable_sequence[dv_idx]["name"]
                if sv_name == dv_name:
                    dfg_edges.append((sv_idx, dv_idx))
                    
    return variable_sequence, list(set(dfg_edges))

def process_dataset(jsonl_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {jsonl_path}...")
    
    count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            filename = item['filename']
            source = item['source_code']
            
            try:
                vars_seq, edges = extract_dataflow(source)
                res = {
                    "filename": filename,
                    "variable_sequence": vars_seq,
                    "dfg_edges": edges
                }
                
                # Save as individual json or shard
                out_path = os.path.join(output_dir, f"{filename}.df.json")
                with open(out_path, 'w') as out_f:
                    json.dump(res, out_f)
                count += 1
            except Exception as e:
                # print(f"Error processing {filename}: {e}")
                continue
                
    print(f"Extraction complete. Processed {count} contracts.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    process_dataset(args.jsonl, args.output_dir)
