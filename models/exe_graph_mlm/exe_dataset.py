#!/usr/bin/env python3
"""
Dataset for Exe Graph MLM (Execution Trajectory Prediction).
Loads NPZ shards containing code features + trajectory arrays.
Dynamically masks topological trajectories based on curriculum masking ratios.
"""

import json
import os
import random
import glob
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ExeGraphDataset(Dataset):
    def __init__(
        self,
        npz_dir: str,
        traj_dir: str,
        bucket_dir: str,
        tokenizer,
        max_len: int = 1024,
        num_trajectories: int = 1,          # Multi-path masking
        trajectory_mask_ratio: float = 0.5, # Curriculum parameter
        base_mlm_prob: float = 0.05,        # Baseline noise
        split: str = "train",
        val_frac: float = 0.05,
        test_frac: float = 0.05,
        seed: int = 42,
        mask_dir: str = None,               # Pre-computed mask directory
        source_index: Optional[dict] = None, # Pre-built lazy index
        tokenized_dir: str = None            # Directory with pre-tokenized .npz files
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_trajectories = num_trajectories
        self.trajectory_mask_ratio = max(0.01, min(1.0, trajectory_mask_ratio))
        self.base_mlm_prob = base_mlm_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size

        self.npz_dir = npz_dir
        self.traj_dir = traj_dir
        self.mask_dir = mask_dir
        self.use_precomputed_masks = mask_dir is not None and os.path.isdir(mask_dir or "")
        self.tokenized_dir = tokenized_dir
        self.use_tokenized_cache = tokenized_dir is not None and os.path.isdir(tokenized_dir)

        full_shard_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        shard_files = [sf for sf in full_shard_files if not os.path.basename(sf).startswith("err_")]
        if not shard_files:
            raise FileNotFoundError(f"No shards found in {npz_dir}")

        self.index = []
        for sf in shard_files:
            d = np.load(sf, allow_pickle=True)
            manifest = json.loads(d["manifest"][0])
            for entry in manifest:
                entry_dict = {
                    "shard_path": sf,
                    "traj_path": os.path.join(traj_dir, os.path.basename(sf)),
                    "idx": entry["idx"],
                    "contract_id": entry["contract_id"],
                }
                if self.use_precomputed_masks:
                    entry_dict["mask_path"] = os.path.join(mask_dir, os.path.basename(sf))
                if self.use_tokenized_cache:
                    shard_name = os.path.basename(sf)
                    # Support both .npz and .jsonl source names in token cache
                    token_shard = shard_name if shard_name.endswith(".npz") else shard_name.replace(".jsonl", ".npz")
                    # If the npz_dir contains shredded shards, they might not match 1-1 with merged token shards
                    # But if using merged mode, it works.
                    entry_dict["tokenized_path"] = os.path.join(tokenized_dir, token_shard)
                self.index.append(entry_dict)
            d.close()

        if source_index is not None:
            self.source_index = source_index
        else:
            # Fallback (slow but safe)
            self.source_index = {}
            # ... (kept internal indexing logic if needed or just error)
            pass

        valid = [e for e in self.index if e["contract_id"] in self.source_index]
        self.index = valid

        # Split
        rng = random.Random(seed)
        rng.shuffle(self.index)
        n = len(self.index)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        if split == "test":
            self.index = self.index[:n_test]
        elif split == "val":
            self.index = self.index[n_test:n_test + n_val]
        else:
            self.index = self.index[n_test + n_val:]


        self._shard_cache = {}
        self._traj_cache = {}
        self._mask_cache = {}
        self._max_cached = 2  # Reduced from 32 to prevent OOM with large merged shards

        self._cache_order = []
        self._token_cache = {}
        self._addr_to_token_shard = {}
        
        if self.use_tokenized_cache:
            shard_files = sorted(glob.glob(os.path.join(tokenized_dir, "*.npz")))
            print(f"INFO: Indexing token cache from {len(shard_files)} shards...", flush=True)
            for sf in shard_files:
                try:
                    d = np.load(sf, allow_pickle=True)
                    if "addresses" in d:
                        for addr in d["addresses"]:
                            self._addr_to_token_shard[addr] = sf
                    d.close()
                except Exception as e:
                    print(f"WARNING: Failed to index token shard {sf}: {e}", flush=True)
            print(f"INFO: Token cache indexed {len(self._addr_to_token_shard)} contracts.", flush=True)
            # Read K from first mask shard
            mask_shards = sorted(glob.glob(os.path.join(mask_dir, "*.npz")))
            if mask_shards:
                d = np.load(mask_shards[0], allow_pickle=True)
                self._mask_K = int(d['K'][0]) if 'K' in d else 10
                d.close()
            else:
                self._mask_K = 10
                self.use_precomputed_masks = False
            print(f"INFO: Pre-computed masks enabled (K={self._mask_K}, dir={mask_dir})", flush=True)

    def __len__(self):
        return len(self.index)
        
    def set_curriculum(self, num_trajectories: int, ratio: float):
        """Allows dynamic curriculum updates from the adaptive scheduler."""
        self.num_trajectories = max(1, int(num_trajectories))
        self.trajectory_mask_ratio = max(0.01, min(1.0, ratio))

    def _get_shard(self, shard_path: str, cache: dict):
        if shard_path in cache:
            return cache[shard_path]
        if len(cache) >= self._max_cached:
            # Pop oldest
            oldest = next(iter(cache))
            cache[oldest].close()
            del cache[oldest]
        
        try:
            # Use mmap_mode='r' where possible to save memory (though npz lazy loading helps)
            d = np.load(shard_path, allow_pickle=True)
            cache[shard_path] = d
            # Worker heartbeat
            shard_name = os.path.basename(shard_path)
            print(f"DEBUG: Worker {os.getpid()} loaded shard {shard_name}", flush=True)
            return d

        except Exception:
            return {}

    def _tokenize(self, source_code: str):
        encoded = self.tokenizer(
            source_code,
            add_special_tokens=True,
            truncation=False, # Removed truncation limits!
            return_tensors="pt"
        )
        full_ids = encoded["input_ids"].squeeze(0)
        total_tokens = len(full_ids)
        
        if total_tokens > self.max_len:
            # Dynamic Sliding Sequence Window
            start_idx = random.randint(0, total_tokens - self.max_len)
            end_idx = start_idx + self.max_len
            input_ids = full_ids[start_idx:end_idx]
            attention_mask = torch.ones(self.max_len, dtype=torch.long)
            word_ids = encoded.word_ids(0)[start_idx:end_idx]
        else:
            pad_len = self.max_len - total_tokens
            input_ids = torch.cat([full_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones(total_tokens, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            word_ids = encoded.word_ids(0) + [None] * pad_len
            
        return input_ids, attention_mask, word_ids

    def _process_tokenized(self, full_ids, full_word_ids):
        """Applies sliding window or padding to already-tokenized arrays."""
        total_tokens = len(full_ids)
        
        if total_tokens > self.max_len:
            # Dynamic Sliding Sequence Window
            start_idx = random.randint(0, total_tokens - self.max_len)
            end_idx = start_idx + self.max_len
            input_ids = full_ids[start_idx:end_idx]
            attention_mask = torch.ones(self.max_len, dtype=torch.long)
            word_ids = full_word_ids[start_idx:end_idx]
        else:
            pad_len = self.max_len - total_tokens
            input_ids = torch.cat([full_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([torch.ones(total_tokens, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            word_ids = full_word_ids + [None] * pad_len
            
        return input_ids, attention_mask, word_ids

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        addr = entry["contract_id"]
        
        input_ids, attention_mask, word_ids = None, None, None

        # ── Fast Path: Use Pre-Tokenized Cache ──
        if self.use_tokenized_cache:
            token_path = self._addr_to_token_shard.get(addr)
            if token_path:
                token_shard = self._get_shard(token_path, self._token_cache)
                if token_shard and f"input_ids_{addr}" in token_shard:
                    raw_ids = torch.from_numpy(token_shard[f"input_ids_{addr}"]).long()
                    raw_words = token_shard[f"word_ids_{addr}"].tolist()
                    # Map -1 back to None
                    raw_words = [w if w != -1 else None for w in raw_words]
                    input_ids, attention_mask, word_ids = self._process_tokenized(raw_ids, raw_words)

        # ── Slow Path Fallback: Runtime Tokenization ──
        if input_ids is None:
            bf, offset = self.source_index[addr]
            with open(bf, "rb") as f:
                f.seek(offset)
                line = f.readline()
                rec = json.loads(line.decode("utf-8"))
                source = rec["source_code"]
            input_ids, attention_mask, word_ids = self._tokenize(source)

        
        shard_data = self._get_shard(entry["shard_path"], self._shard_cache)
        # Skip trajectory shard loading when using pre-computed masks
        traj_data = None if self.use_precomputed_masks else self._get_shard(entry["traj_path"], self._traj_cache)
        
        ci = entry["idx"]
        graph_features = np.zeros((self.max_len, 6), dtype=np.float32)
        mlm_labels = torch.full((self.max_len,), -100, dtype=torch.long)
        
        if not shard_data or (not self.use_precomputed_masks and not traj_data):
            return {
                "input_ids": input_ids, "attention_mask": attention_mask,
                "graph_features": torch.zeros((self.max_len, 6), dtype=torch.float32),
                "mlm_labels": mlm_labels
            }

        node_features = shard_data.get(f"node_features_{ci}", np.zeros((1, 6)))
        token_to_node = shard_data.get(f"token_to_node_{ci}", np.zeros(1))
        trajectories = traj_data.get(f"trajectories_{ci}", np.empty(0)) if traj_data else np.empty(0)
        
        T_raw = len(token_to_node)
        
        # Populate token level graph features
        for i in range(min(self.max_len, len(word_ids))):
            word_idx = word_ids[i]
            if word_idx is not None and word_idx < T_raw:
                nid = token_to_node[word_idx]
                if nid >= 0 and nid < len(node_features):
                    graph_features[i] = node_features[nid]

        # Trajectory Masking Logic
        nodes_to_mask = set()
        
        # ── Fast path: use pre-computed masks ──
        if self.use_precomputed_masks and "mask_path" in entry:
            mask_data = self._get_shard(entry["mask_path"], self._mask_cache)
            if mask_data:
                k = random.randint(0, self._mask_K - 1)
                mask_key = f"mask_nodes_{ci}_{k}"
                if mask_key in mask_data:
                    mask_nodes = mask_data[mask_key]
                    if self.trajectory_mask_ratio < 1.0:
                        # Sub-sample the pre-computed mask set
                        n_keep = max(1, int(len(mask_nodes) * self.trajectory_mask_ratio))
                        if n_keep < len(mask_nodes):
                            mask_nodes = np.random.choice(mask_nodes, n_keep, replace=False)
                    nodes_to_mask = set(mask_nodes.tolist())
        # ── Fallback: runtime trajectory sampling ──
        elif len(trajectories) > 0:
            n_sample = min(self.num_trajectories, len(trajectories))
            chosen_trajs = random.sample(list(trajectories), n_sample)
            
            for traj in chosen_trajs:
                if len(traj) > 0:
                    num_to_mask = max(1, int(len(traj) * self.trajectory_mask_ratio))
                    if self.trajectory_mask_ratio < 1.0:
                        chosen_subset = random.sample(list(traj), num_to_mask)
                        nodes_to_mask.update(chosen_subset)
                    else:
                        nodes_to_mask.update(traj)

        # Baseline random masking (so standard syntax structures still provide some gradients)
        all_maskable = [i for i, wid in enumerate(word_ids) if wid is not None and input_ids[i] not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]
        
        for i in all_maskable:
            word_idx = word_ids[i]
            if word_idx is not None and word_idx < T_raw:
                nid = token_to_node[word_idx]
                prob = random.random()
                
                # Apply trajectory deterministic masking OR baseline random masking
                if (nid in nodes_to_mask) or (prob < self.base_mlm_prob):
                    mlm_labels[i] = input_ids[i]
                    
                    # 80/10/10 BERT masking rule
                    r = random.random()
                    if r < 0.8:
                        input_ids[i] = self.mask_token_id
                    elif r < 0.9:
                        input_ids[i] = random.randint(0, self.vocab_size - 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_features": torch.from_numpy(graph_features),
            "mlm_labels": mlm_labels,
        }

def create_exe_dataloaders(npz_dir, traj_dir, bucket_dir, tokenizer, batch_size=4, max_len=1024, init_mask_ratio=0.3, mask_dir=None, tokenized_dir=None):
    # Build robust lazy source code index once
    source_index = {}
    bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "*.jsonl")))
    print(f"INFO: Building source index from {len(bucket_files)} buckets...", flush=True)
    for bf in bucket_files:
        with open(bf, "rb") as f:
            offset = 0
            for line in f:
                try:
                    rec = json.loads(line.decode("utf-8"))
                    addr = rec.get("contract_address")
                    if addr:
                        source_index[addr] = (bf, offset)
                except: pass
                offset += len(line)
    print(f"INFO: Indexed {len(source_index)} contracts.", flush=True)

    train_ds = ExeGraphDataset(npz_dir, traj_dir, bucket_dir, tokenizer, split="train", max_len=max_len, trajectory_mask_ratio=init_mask_ratio, mask_dir=mask_dir, source_index=source_index, tokenized_dir=tokenized_dir)
    val_ds = ExeGraphDataset(npz_dir, traj_dir, bucket_dir, tokenizer, split="val", max_len=max_len, trajectory_mask_ratio=1.0, mask_dir=mask_dir, source_index=source_index, tokenized_dir=tokenized_dir)

    
    from torch.utils.data import Sampler
    class ShardSampler(Sampler):
        """Forces PyTorch to sample contracts by shard, eliminating 99% of I/O thrashing np.load() misses."""
        def __init__(self, data_source):
            self.data_source = data_source
            self.shard_to_idx = {}
            for i, entry in enumerate(data_source.index):
                shard = entry['shard_path']
                if shard not in self.shard_to_idx:
                    self.shard_to_idx[shard] = []
                self.shard_to_idx[shard].append(i)
        def __iter__(self):
            shards = list(self.shard_to_idx.keys())
            random.shuffle(shards)
            for shard in shards:
                indices = self.shard_to_idx[shard].copy()
                random.shuffle(indices)
                for idx in indices:
                    yield idx
        def __len__(self):
            return len(self.data_source)

    train_sampler = ShardSampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, 
        pin_memory=True, num_workers=4, persistent_workers=True
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader, train_ds

