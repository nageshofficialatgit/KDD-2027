import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import glob

# Ensure Implementation/ is in path for model imports
sys.path.append("./Implementation")

class SupervisedVulnerabilityDataset(Dataset):
    """
    Unified dataset for supervised vulnerability detection.
    Supports both text-only (baselines) and graph-augmented (WIRE) inputs.
    """
    def __init__(self, labels_path, tokenizer, max_len=2048, npz_dir=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print(f"Loading labels from {labels_path}...")
        with open(labels_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.vuln_types = [
            "Reentrancy", "Integer_Overflow_Underflow", "Access_Control", 
            "Timestamp_Dependence", "Transaction_Ordering_Dependence_Front_Running", 
            "tx_origin", "Bad_Randomness", "Unchecked_Low_Level_Calls"
        ]
        
        # Graph feature indexing (optional)
        self.npz_index = {} # contract_id -> (shard_path, idx_in_shard)
        if npz_dir:
            print(f"Indexing graph shards in {npz_dir}...")
            shard_files = sorted(glob.glob(os.path.join(npz_dir, "v*_shard_*.npz")))
            for sf in shard_files:
                d = np.load(sf, allow_pickle=True)
                manifest = json.loads(d["manifest"][0])
                for entry in manifest:
                    self.npz_index[entry["contract_id"]] = (sf, entry["idx"])
                d.close()
            print(f"Indexed {len(self.npz_index)} graph-ready contracts.")

    def __len__(self):
        return len(self.data)

    def _get_graph_features(self, contract_id, encoding):
        """Re-implementing token-to-node mapping logic from GraphRoPEDataset"""
        # (Simplified version for evaluation purposes)
        # In a real scenario, we'd reuse the GraphRoPEDataset logic exactly.
        # Here we initialize with zeros (Null coordinates)
        graph_feats = np.zeros((self.max_len, 6), dtype=np.float32)
        
        if contract_id not in self.npz_index:
            return graph_feats
            
        shard_path, ci = self.npz_index[contract_id]
        shard = np.load(shard_path, allow_pickle=True)
        
        node_features = shard[f"node_features_{ci}"]
        token_to_node = shard[f"token_to_node_{ci}"]
        
        # Handle NaN/Inf
        if not np.isfinite(node_features).all():
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            
        word_ids = encoding.word_ids(0)
        T_raw = len(token_to_node)
        
        for i in range(min(self.max_len, len(word_ids))):
            word_idx = word_ids[i]
            if word_idx is not None and word_idx < T_raw:
                nid = token_to_node[word_idx]
                if nid >= 0:
                    graph_feats[i] = node_features[nid]
        
        shard.close()
        return graph_feats

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item['source_code']
        labels = [item['labels'][v] for v in self.vuln_types]
        contract_id = item['filename']
        
        encoding = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True # Important for word_ids
        )
        
        # Get graph features if npz_dir was provided
        # Note: word_ids needs the encoding object
        res = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }
        
        if self.npz_index:
             # We need to manually call word_ids(0) as the pt return_tensors changes the object
             # Or use the encoding object directly before squeezed
             g_feats = self._get_graph_features(contract_id, encoding)
             res["graph_features"] = torch.from_numpy(g_feats).float()
             
        return res

def get_hash_splits(labels_path, seed=42):
    """
    Split labels.jsonl using cryptographic hashes to prevent cross-dataset leakage.
    """
    import random
    rng = random.Random(seed)
    
    hash_to_indices = {}
    with open(labels_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            h = item['stripped_hash']
            if h not in hash_to_indices:
                hash_to_indices[h] = []
            hash_to_indices[h].append(i)
    
    unique_hashes = list(hash_to_indices.keys())
    rng.shuffle(unique_hashes)
    
    n = len(unique_hashes)
    n_test = int(n * 0.15)
    n_val = int(n * 0.10)
    
    test_hashes = unique_hashes[:n_test]
    val_hashes = unique_hashes[n_test:n_test+n_val]
    train_hashes = unique_hashes[n_test+n_val:]
    
    def get_indices(hashes):
        indices = []
        for h in hashes:
            indices.extend(hash_to_indices[h])
        return indices

    return get_indices(train_hashes), get_indices(val_hashes), get_indices(test_hashes)
