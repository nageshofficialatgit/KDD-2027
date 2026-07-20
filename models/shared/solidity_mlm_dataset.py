#!/usr/bin/env python3
"""
Vanilla ModernBERT Solidity MLM Pretraining — Dataset.

Reads source code from bucket JSONL files (same data as Graph-RoPE),
tokenizes, and applies standard MLM masking. No graph features.
"""

import json
import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SolidityMLMDataset(Dataset):
    """
    Simple MLM dataset for Solidity source code.
    Reads from the same bucket JSONL files used by Graph-RoPE pretraining.
    """

    def __init__(
        self,
        contracts: list,
        tokenizer,
        max_len: int = 2048,
        mlm_prob: float = 0.30,
    ):
        super().__init__()
        self.contracts = contracts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        source = self.contracts[idx]

        # Tokenize
        enc = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)        # [max_len]
        attention_mask = enc["attention_mask"].squeeze(0)  # [max_len]

        # MLM masking (same 30% rate as Graph-RoPE pretraining)
        mlm_labels = input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_prob)

        # Don't mask special tokens or padding
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in self.tokenizer.all_special_ids:
            special |= (input_ids == sid)
        special |= (attention_mask == 0)
        probability_matrix.masked_fill_(special, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100  # only compute loss on masked

        # 80% [MASK], 10% random, 10% keep original
        indices_replace = masked_indices & (torch.rand(input_ids.shape) < 0.8)
        input_ids[indices_replace] = self.mask_token_id

        indices_random = masked_indices & ~indices_replace & (torch.rand(input_ids.shape) < 0.5)
        input_ids[indices_random] = torch.randint(self.vocab_size, input_ids.shape)[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": mlm_labels,
        }


def create_vanilla_dataloaders(
    bucket_dir: str,
    tokenizer,
    max_len: int = 2048,
    batch_size: int = 16,
    mlm_prob: float = 0.30,
    max_contracts: int = 0,
    num_workers: int = 4,
    seed: int = 42,
    npz_dir: str = "",
):
    """Create train/val/test dataloaders for vanilla MLM pretraining."""
    
    valid_ids = None
    if npz_dir and os.path.exists(npz_dir):
        import numpy as np
        print(f"Reading valid contract IDs from NPZ manifests in {npz_dir}...", flush=True)
        valid_ids = set()
        for sf in sorted(glob.glob(os.path.join(npz_dir, "v*_shard_*.npz"))):
            try:
                d = np.load(sf, allow_pickle=True)
                manifest = json.loads(d["manifest"][0])
                for e in manifest:
                    valid_ids.add(e["contract_id"])
                d.close()
            except Exception as e:
                print(f"Warning: could not read manifest from {sf}: {e}", flush=True)
        print(f"Found {len(valid_ids):,} valid contract IDs in NPZ shards", flush=True)

    # Load ALL source code ONCE
    print(f"Loading source code from {bucket_dir}...", flush=True)
    all_contracts = []
    bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "*.jsonl")))
    if not bucket_files:
        raise FileNotFoundError(f"No .jsonl files in {bucket_dir}")
    for bf in bucket_files:
        with open(bf) as f:
            for line in f:
                rec = json.loads(line)
                addr = rec.get("contract_address", "")
                src = rec.get("source_code", "")
                if src and len(src) > 50:
                    if valid_ids is not None and addr not in valid_ids:
                        continue
                    all_contracts.append(src)
    print(f"Total contracts loaded: {len(all_contracts):,}", flush=True)

    # Cap total contracts if requested
    if max_contracts > 0 and len(all_contracts) > max_contracts:
        rng_cap = random.Random(seed)
        rng_cap.shuffle(all_contracts)
        all_contracts = all_contracts[:max_contracts]
        print(f"Capped to {len(all_contracts):,} contracts (matching WIRE/baseline)", flush=True)

    # Split by indices (not copies of the list)
    rng = random.Random(seed)
    indices = list(range(len(all_contracts)))
    rng.shuffle(indices)
    n = len(indices)
    n_test = int(n * 0.05)
    n_val = int(n * 0.05)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}", flush=True)

    common = dict(tokenizer=tokenizer, max_len=max_len, mlm_prob=mlm_prob)
    train_ds = SolidityMLMDataset([all_contracts[i] for i in train_idx], **common)
    val_ds = SolidityMLMDataset([all_contracts[i] for i in val_idx], **common)
    test_ds = SolidityMLMDataset([all_contracts[i] for i in test_idx], **common)

    # Free the master list
    del all_contracts

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader

