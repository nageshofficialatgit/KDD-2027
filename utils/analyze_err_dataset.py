#!/usr/bin/env python3
import os
import numpy as np
import glob
import json
import argparse
from tqdm import tqdm

def analyze_err_dataset(feat_dir):
    print(f"Scanning directory: {feat_dir}")
    shard_files = sorted(glob.glob(os.path.join(feat_dir, "v*_shard_*.npz")))
    
    stats = {
        "total_contracts": 0,
        "total_err_pairs": 0,
        "err_values": [],
        "nan_count_features": 0,
        "inf_count_features": 0,
        "nan_count_err": 0,
        "inf_count_err": 0,
        "feat_min": None,
        "feat_max": None,
        "feat_sums": np.zeros(6),
        "feat_sq_sums": np.zeros(6),
        "total_nodes": 0
    }

    pbar = tqdm(shard_files, desc="Analyzing shards")
    for sf in pbar:
        try:
            data = np.load(sf, allow_pickle=True)
            err_path = os.path.join(feat_dir, f"err_{os.path.basename(sf)}")
            err_data_loaded = None
            if os.path.exists(err_path):
                err_data_loaded = np.load(err_path, allow_pickle=True)
            
            manifest_key = "manifest" if "manifest" in data else None
            if manifest_key:
                manifest = json.loads(data[manifest_key][0])
                for entry in manifest:
                    ci = entry["idx"]
                    cid = entry["contract_id"]
                    stats["total_contracts"] += 1
                    
                    # 1. Analyze Node Features
                    feat_key = f"node_features_{ci}"
                    if feat_key in data:
                        feats = data[feat_key]
                        stats["total_nodes"] += feats.shape[0]
                        
                        # Count NaNs/Infs
                        nans = np.isnan(feats).sum()
                        infs = np.isinf(feats).sum()
                        stats["nan_count_features"] += nans
                        stats["inf_count_features"] += infs
                        
                        # Accumulate for mean/std
                        # Filter to finite for min/max
                        finite_feats = feats[np.isfinite(feats).all(axis=1)]
                        if finite_feats.size > 0:
                            f_min = finite_feats.min(axis=0)
                            f_max = finite_feats.max(axis=0)
                            if stats["feat_min"] is None:
                                stats["feat_min"] = f_min
                                stats["feat_max"] = f_max
                            else:
                                stats["feat_min"] = np.minimum(stats["feat_min"], f_min)
                                stats["feat_max"] = np.maximum(stats["feat_max"], f_max)
                            
                            stats["feat_sums"] += finite_feats.sum(axis=0)
                            stats["feat_sq_sums"] += (finite_feats**2).sum(axis=0)

                    # 2. Analyze ERR Data
                    err_key = f"err_pairs_{ci}"
                    if err_data_loaded and err_key in err_data_loaded:
                        err_vals = err_data_loaded[err_key]
                        if err_vals.ndim == 2 and err_vals.shape[1] >= 3:
                            vals = err_vals[:, 2]
                            stats["total_err_pairs"] += vals.size
                            
                            # NaNs/Infs
                            stats["nan_count_err"] += np.isnan(vals).sum()
                            stats["inf_count_err"] += np.isinf(vals).sum()
                            
                            # Store sample for median (subsample if too large)
                            if len(stats["err_values"]) < 1000000:
                                stats["err_values"].extend(vals[np.isfinite(vals)].tolist())
            
            data.close()
            if err_data_loaded: err_data_loaded.close()
            
        except Exception as e:
            print(f"Error processing {sf}: {e}")

    # Final calculations
    print("\n" + "="*50)
    print("ERR DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Total Contracts:    {stats['total_contracts']:,}")
    print(f"Total Nodes:        {stats['total_nodes']:,}")
    print(f"Total ERR Pairs:    {stats['total_err_pairs']:,}")
    
    print("\n--- Stability ---")
    print(f"NaNs in Features:   {stats['nan_count_features']:,}")
    print(f"Infs in Features:   {stats['inf_count_features']:,}")
    print(f"NaNs in ERR:        {stats['nan_count_err']:,}")
    print(f"Infs in ERR:        {stats['inf_count_err']:,}")
    
    if stats["total_err_pairs"] > 0 and len(stats["err_values"]) > 0:
        err_arr = np.array(stats["err_values"])
        print("\n--- ERR Value Distribution ---")
        print(f"Mean ERR:           {np.mean(err_arr):.4f}")
        print(f"Median ERR:         {np.median(err_arr):.4f}")
        print(f"Min ERR:            {np.min(err_arr):.4f}")
        print(f"Max ERR:            {np.max(err_arr):.4f}")
        print(f"P5 ERR:             {np.percentile(err_arr, 5):.4f}")
        print(f"P95 ERR:            {np.percentile(err_arr, 95):.4f}")

    if stats["total_nodes"] > 0:
        mean = stats["feat_sums"] / stats["total_nodes"]
        # var = E[X^2] - (E[X])^2
        var = (stats["feat_sq_sums"] / stats["total_nodes"]) - (mean**2)
        std = np.sqrt(np.maximum(var, 0))
        
        print("\n--- Feature Matrix Statistics (6 dims) ---")
        dims = ["poisson", "d_scale1", "d_scale2", "d_scale3", "d_scale4", "d_scale5"]
        header = f"{'Dimension':<12} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}"
        print(header)
        print("-" * len(header))
        for i, name in enumerate(dims):
            print(f"{name:<12} | {mean[i]:<10.4f} | {std[i]:<10.4f} | {stats['feat_min'][i]:<10.4f} | {stats['feat_max'][i]:<10.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", default="./model/cpg_features")
    args = parser.parse_args()
    analyze_err_dataset(args.feat_dir)
