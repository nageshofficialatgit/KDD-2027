#!/usr/bin/env python3
"""
extract_trajectories.py
-----------------------
Pre-computes execution trajectories from CPG graphs for Path-Masked Language Modeling (PMLM).
Uses multi-processing to parse shards rapidly.
Extracts strict vulnerability-semantic paths: [Check / Entry] -> [State Write / External Call]
"""
import glob
import os
import json
import argparse
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp

def _reconstruct_sparse(data_dict, shape_val=None):
    if not isinstance(data_dict, dict) and not hasattr(data_dict, 'tolist'):
        # Usually serialized as dict
        data_dict = data_dict.item()
    return sp.coo_matrix(
        (data_dict['data'], (data_dict['row'], data_dict['col'])), 
        shape=data_dict.get('shape', shape_val)
    ).tocsr()

def extract_paths_contract(idx, graph_data):
    """
    Extract execution trajectories for a single contract inside a shard.
    Returns a list of node arrays representing execution traces.
    """
    try:
        # Load matrices
        N = len(graph_data[f"node_opcodes_{idx}"])
        A_cfg = sp.coo_matrix(
            (graph_data[f"graph_cfg_data_{idx}"], 
             (graph_data[f"graph_cfg_row_{idx}"], graph_data[f"graph_cfg_col_{idx}"])),
            shape=tuple(graph_data[f"graph_cfg_shape_{idx}"])
        ).tocsr()
        
        A_dfg = sp.coo_matrix(
            (graph_data[f"graph_dfg_data_{idx}"], 
             (graph_data[f"graph_dfg_row_{idx}"], graph_data[f"graph_dfg_col_{idx}"])),
            shape=tuple(graph_data[f"graph_dfg_shape_{idx}"])
        ).tocsr()
        
        A_call = sp.csr_matrix((N, N))
        if f"graph_call_row_{idx}" in graph_data:
            cshape = tuple(graph_data[f"graph_call_shape_{idx}"]) if len(graph_data[f"graph_call_shape_{idx}"]) == 2 else (N, N)
            A_call = sp.coo_matrix(
                (graph_data[f"graph_call_data_{idx}"], 
                 (graph_data[f"graph_call_row_{idx}"], graph_data[f"graph_call_col_{idx}"])),
                shape=cshape
            ).tocsr()

        A_combo = A_cfg + A_dfg + A_call

        node_flags = graph_data[f"node_flags_{idx}"]
        node_opcodes = graph_data[f"node_opcodes_{idx}"]
        
        # Sinks: External calls or state writes
        is_ext_call = node_flags[:, 2] == 1
        is_state_write = node_flags[:, 3] == 1
        sinks = np.where(is_ext_call | is_state_write)[0]
        
        # Sources: IF condition, Require, Entry
        sources = []
        for i, op in enumerate(node_opcodes):
            sop = str(op)
            if 'IF' in sop or 'ENTRY' in sop or 'THROW' in sop: # Slither requires usually compile with THROW
                sources.append(i)
        
        if not sources or not sinks.size:
            return []
            
        traces = []
        distances, predecessors = sp.csgraph.shortest_path(
            A_combo, directed=False, return_predecessors=True, unweighted=True, indices=sinks
        )
        
        for sink_idx_local, sink in enumerate(sinks):
            for src in sources:
                dist = distances[sink_idx_local, src]
                if dist > 0 and dist < 12 and not np.isinf(dist): 
                    # Reconstruct path
                    path = []
                    curr = src
                    while curr != sink and curr != -9999:
                        path.append(curr)
                        curr = predecessors[sink_idx_local, curr]
                        if len(path) > 15: # safety cutoff
                            break
                    path.append(sink)
                    if len(path) > 2:
                        traces.append(np.array(path, dtype=np.int32))
                        
        # Deduplicate
        unique_traces = []
        seen = set()
        for t in traces:
            fst = tuple(t.tolist())
            if fst not in seen:
                seen.add(fst)
                unique_traces.append(t)
                
        return unique_traces
        
    except Exception as e:
        return []

def worker(shard_path, out_dir):
    try:
        data = np.load(shard_path, allow_pickle=True)
        manifest = json.loads(data['manifest'][0])
        
        out_data = {}
        out_manifest = []
        
        for entry in manifest:
            idx = entry['idx']
            cid = entry['contract_id']
            traces = extract_paths_contract(idx, data)
            # Encode list of numpy arrays into single object array
            if traces:
                out_data[f"trajectories_{idx}"] = np.empty(len(traces), dtype=object)
                for i, tr in enumerate(traces):
                    out_data[f"trajectories_{idx}"][i] = tr
            else:
                out_data[f"trajectories_{idx}"] = np.empty(0, dtype=object)
                
            out_manifest.append(entry)
            
        out_data['manifest'] = np.array([json.dumps(out_manifest)], dtype=object)
        
        shard_name = os.path.basename(shard_path)
        out_file = os.path.join(out_dir, shard_name)
        np.savez(out_file, **out_data)
        data.close()
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed {shard_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default="../../model/cpg_graphs")
    parser.add_argument("--out-dir", default="../../model/cpg_trajectories")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    graph_dir_full = os.path.join(project_root, args.graph_dir.lstrip("../").strip("/"))
    out_dir_full = os.path.join(project_root, args.out_dir.lstrip("../").strip("/"))
    
    os.makedirs(out_dir_full, exist_ok=True)

    
    shards = glob.glob(os.path.join(graph_dir_full, "*.npz"))
    print(f"Found {len(shards)} shards in {graph_dir_full}")
    
    pool_args = [(s, out_dir_full) for s in shards]
    
    with multiprocessing.Pool(args.workers) as pool:
        for _ in tqdm(pool.starmap(worker, pool_args), total=len(shards), desc="Extracting Traces"):
            pass
            
    print("Trajectory Extraction Complete.")

if __name__ == "__main__":
    main()
