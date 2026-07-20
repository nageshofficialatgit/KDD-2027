import os
import glob
import numpy as np
from tqdm import tqdm

def precompute_metp_shards(graph_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    shard_files = sorted(glob.glob(os.path.join(graph_dir, "v*_shard_*.npz")))
    print(f"Found {len(shard_files)} shards to process in {graph_dir}")
    
    for shard_path in tqdm(shard_files, desc="Precomputing METP edges"):
        shard_name = os.path.basename(shard_path)
        out_path = os.path.join(out_dir, shard_name)
        
        # Skip if already precomputed
        if os.path.exists(out_path):
            continue
            
        data = np.load(shard_path, allow_pickle=True)
        out_dict = {}
        
        # Find all contract indices in this shard
        # A shard contains keys like graph_cfg_row_0, graph_dfg_col_1, etc.
        indices = set()
        for key in data.keys():
            if key.startswith("graph_"):
                idx = key.split("_")[-1]
                indices.add(int(idx))
                
        for ci in indices:
            edge_list = []
            
            # Label 0: CFG
            if f"graph_cfg_row_{ci}" in data:
                r = data[f"graph_cfg_row_{ci}"]
                c = data[f"graph_cfg_col_{ci}"]
                for u, v in zip(r, c):
                    edge_list.append((u, v, 0))
                    
            # Label 1: DFG
            if f"graph_dfg_row_{ci}" in data:
                r = data[f"graph_dfg_row_{ci}"]
                c = data[f"graph_dfg_col_{ci}"]
                for u, v in zip(r, c):
                    edge_list.append((u, v, 1))
                    
            # Label 2: Call Graph
            if f"graph_call_row_{ci}" in data:
                r = data[f"graph_call_row_{ci}"]
                c = data[f"graph_call_col_{ci}"]
                for u, v in zip(r, c):
                    edge_list.append((u, v, 2))
            
            if len(edge_list) > 0:
                # Convert to numpy array of shape (E, 3)
                # We use int32 to save disk space and RAM, since node indices rarely exceed 100,000
                edges_arr = np.array(edge_list, dtype=np.int32)
            else:
                edges_arr = np.zeros((0, 3), dtype=np.int32)
                
            out_dict[f"edges_{ci}"] = edges_arr
            
        # Save the new optimized shard
        np.savez_compressed(out_path, **out_dict)

if __name__ == "__main__":
    GRAPH_DIR = "../model/cpg_graphs"
    OUT_DIR = "../model/metp_edges"
    precompute_metp_shards(GRAPH_DIR, OUT_DIR)
    print("Precomputation finished!")
