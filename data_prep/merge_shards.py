import numpy as np
import os
import json
import glob
from tqdm import tqdm
import argparse

def merge_shards(src_feat_dir, src_mask_dir, dest_feat_dir, dest_mask_dir, shards_per_merged=20):
    os.makedirs(dest_feat_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    feat_shards = sorted(glob.glob(os.path.join(src_feat_dir, "*.npz")))
    feat_shards = [f for f in feat_shards if not os.path.basename(f).startswith("err_")]

    print(f"Found {len(feat_shards)} feature shards.")

    for i in range(0, len(feat_shards), shards_per_merged):
        batch = feat_shards[i:i + shards_per_merged]
        merged_idx = i // shards_per_merged
        
        merged_feat_data = {}
        merged_mask_data = {}
        
        new_feat_manifest = []
        new_mask_manifest = []
        
        global_count = 0
        
        for sf in tqdm(batch, desc=f"Merging batch {merged_idx}", leave=False):
            shard_name = os.path.basename(sf)
            mask_path = os.path.join(src_mask_dir, shard_name)
            
            # Load feature shard
            f_data = np.load(sf, allow_pickle=True)
            f_manifest = json.loads(f_data["manifest"][0])
            
            # Load mask shard if exists
            m_data = None
            if os.path.exists(mask_path):
                m_data = np.load(mask_path, allow_pickle=True)
                m_K = int(m_data['K'][0]) if 'K' in m_data else 10
            
            for entry in f_manifest:
                old_ci = entry["idx"]
                new_ci = global_count
                
                # Copy features
                feat_key = f"node_features_{old_ci}"
                map_key = f"token_to_node_{old_ci}"
                
                if feat_key in f_data:
                    merged_feat_data[f"node_features_{new_ci}"] = f_data[feat_key]
                if map_key in f_data:
                    merged_feat_data[f"token_to_node_{new_ci}"] = f_data[map_key]
                
                # Copy masks if available
                if m_data:
                    found_masks = False
                    for k in range(m_K):
                        mask_key = f"mask_nodes_{old_ci}_{k}"
                        if mask_key in m_data:
                            merged_mask_data[f"mask_nodes_{new_ci}_{k}"] = m_data[mask_key]
                            found_masks = True
                    
                # Update manifest entry
                new_entry = entry.copy()
                new_entry["idx"] = new_ci
                new_feat_manifest.append(new_entry)
                new_mask_manifest.append(new_entry)
                
                global_count += 1
            
            f_data.close()
            if m_data:
                m_data.close()

        # Save merged shards
        import shutil
        _, _, free = shutil.disk_usage("/")
        if free < 2 * 1024 * 1024 * 1024: # 2GB
            print("CRITICAL: Disk full. Stopping.")
            break

        feat_out = os.path.join(dest_feat_dir, f"merged_shard_{merged_idx:04d}.npz")
        merged_feat_data["manifest"] = np.array([json.dumps(new_feat_manifest)], dtype=object)
        print(f"Saving {feat_out}...")
        np.savez(feat_out, **merged_feat_data)

        if merged_mask_data:
            mask_out = os.path.join(dest_mask_dir, f"merged_shard_{merged_idx:04d}.npz")
            merged_mask_data["manifest"] = np.array([json.dumps(new_mask_manifest)], dtype=object)
            merged_mask_data["K"] = np.array([m_K], dtype=np.int32)
            print(f"Saving {mask_out}...")
            np.savez(mask_out, **merged_mask_data)


    print("Merging complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-feat", default="model/cpg_features")
    parser.add_argument("--src-mask", default="model/cpg_masks")
    parser.add_argument("--dest-feat", default="model/cpg_features_merged")
    parser.add_argument("--dest-mask", default="model/cpg_masks_merged")
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    
    merge_shards(args.src_feat, args.src_mask, args.dest_feat, args.dest_mask, args.batch_size)
