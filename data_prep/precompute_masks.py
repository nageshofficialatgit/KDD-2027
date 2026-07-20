#!/usr/bin/env python3
"""
precompute_masks.py
-------------------
Pre-computes trajectory mask node-sets for Exe-MLM training.
For each contract, generates K random trajectory samplings and saves the
resulting node-index sets. At training time, the dataset simply loads
one of the K pre-computed sets instead of doing expensive trajectory
sampling + set construction on every __getitem__ call.

Output format per contract in NPZ:
    mask_nodes_{idx}_{k} : np.int32 array of unique node indices to mask
    (k = 0..K-1)
"""
import glob
import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing

def precompute_contract(idx, trajectories, T, R, K):
    """Generate K random mask node-sets for a single contract."""
    n_trajs = len(trajectories)
    if n_trajs == 0:
        return [np.array([], dtype=np.int32)] * K

    # Convert once (the expensive part we're eliminating from training)
    traj_list = list(trajectories)
    
    results = []
    for _ in range(K):
        n_sample = min(T, n_trajs)
        chosen_trajs = random.sample(traj_list, n_sample)
        
        nodes_to_mask = set()
        for traj in chosen_trajs:
            if hasattr(traj, '__len__') and len(traj) > 0:
                if R < 1.0:
                    num_to_mask = max(1, int(len(traj) * R))
                    chosen_subset = random.sample(list(traj), num_to_mask)
                    nodes_to_mask.update(chosen_subset)
                else:
                    nodes_to_mask.update(traj)
        
        results.append(np.array(sorted(nodes_to_mask), dtype=np.int32))
    
    return results

def worker(args):
    """Process a single trajectory shard."""
    shard_path, out_dir, T, R, K = args
    shard_name = os.path.basename(shard_path)
    out_path = os.path.join(out_dir, shard_name)
    
    # Skip if already processed
    if os.path.exists(out_path):
        return f"SKIP {shard_name}"
    
    try:
        data = np.load(shard_path, allow_pickle=True)
        manifest = json.loads(data['manifest'][0])
        
        out_data = {}
        out_manifest = []
        
        for entry in manifest:
            idx = entry['idx']
            key = f"trajectories_{idx}"
            trajectories = data.get(key, np.empty(0))
            
            masks = precompute_contract(idx, trajectories, T, R, K)
            for k, mask in enumerate(masks):
                out_data[f"mask_nodes_{idx}_{k}"] = mask
            
            out_manifest.append(entry)
        
        out_data['manifest'] = np.array([json.dumps(out_manifest)], dtype=object)
        out_data['K'] = np.array([K], dtype=np.int32)
        out_data['T'] = np.array([T], dtype=np.int32)
        
        np.savez(out_path, **out_data)
        data.close()
        return f"OK {shard_name} ({len(manifest)} contracts)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL {shard_name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Pre-compute trajectory mask node-sets")
    parser.add_argument("--traj-dir", default="model/cpg_trajectories",
                        help="Directory containing trajectory NPZ shards")
    parser.add_argument("--out-dir", default="model/cpg_masks",
                        help="Output directory for pre-computed mask shards")
    parser.add_argument("--T", type=int, default=500,
                        help="Number of trajectories to sample per mask")
    parser.add_argument("--R", type=float, default=1.0,
                        help="Mask ratio (fraction of nodes per path to mask)")
    parser.add_argument("--K", type=int, default=10,
                        help="Number of random mask variants per contract")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    # Script is at Implementation/data_prep/ → project root is ../../..
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    traj_dir = os.path.join(project_root, args.traj_dir)
    out_dir = os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    shards = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
    print(f"Found {len(shards)} trajectory shards in {traj_dir}")
    print(f"Pre-computing T={args.T}, R={args.R}, K={args.K} masks per contract")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}")
    print()
    
    pool_args = [(s, out_dir, args.T, args.R, args.K) for s in shards]
    
    ok = 0
    skip = 0
    fail = 0
    with multiprocessing.Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, pool_args), total=len(shards), desc="Pre-computing masks"):
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                print(f"  {result}")
    
    print(f"\nDone: {ok} processed, {skip} skipped, {fail} failed")

if __name__ == "__main__":
    main()
