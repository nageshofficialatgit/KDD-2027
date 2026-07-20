#!/usr/bin/env python3
"""
Dataset for Graph-RoPE multi-task pretraining.

Loads NPZ feature shards + source code from bucket JSONL files.
Produces batches with:
  - Tokenized source code (ModernBERT tokenizer)
  - Graph features per subword token (6D)
  - MLM targets (15% masking)
  - Node type labels (derived from graph structure)
  - Edge prediction pairs + labels
"""

import json
import math
import os
import random
import glob
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.baseline.graph_rope import (
    DEFAULT_FEAT_P5,
    DEFAULT_FEAT_P95,
)

print(">>> BTP_TELEMETRY: models/graph_rope_dataset.py loaded with TFR normalization [0, 1]")


# ── Topological Role Prediction (TRP) [formerly Node Type] ─────────────────
# Inspiration: GraphCodeBERT's "Node Alignment"
# Goal: Predict structural flow roles (Upstream, Downstream, Hub) from topology.
# We derive these labels from Poisson potential (flow depth) and Diffusion (centrality).
# Security semantics (Source/Sink) are reserved for supervised fine-tuning.
# ── Node type classification (from Poisson potential + connectivity) ─────
# Since the NPZ doesn't store explicit node types, we derive them from
# the graph features themselves. This is a self-supervised task.
# Bins for Poisson potential → structural role:
#   High Poisson (>  p75):  SOURCE   (entry-like, upstream)
#   Low  Poisson (< -p75):  SINK     (exit-like, downstream)
#   Otherwise:              MIDDLE   (intermediate)
# We also consider connectivity via diffusion features:
#   d_scale1 > median:      HUB      (highly connected locally)
#   d_scale5 < p25:         ISOLATED (weakly connected globally)
NUM_NODE_TYPES = 5  # SOURCE, SINK, MIDDLE, HUB, ISOLATED
# These are soft labels derived from features — the model must learn
# to decode structural role from rotary encoding.


class GraphRoPEDataset(Dataset):
    """
    PyTorch Dataset for Graph-RoPE pretraining.

    Each item returns a dict with:
        input_ids:      [max_len]     int64   tokenized source code
        attention_mask: [max_len]     int64   1 for real tokens, 0 for padding
        graph_features: [max_len, 6]  float32 per-subword-token graph features
        mlm_labels:     [max_len]     int64   -100 for non-masked, token_id for masked
        node_type_labels: [max_len]   int64   0-4 node type, -100 for pad
        edge_pairs:     [n_pairs, 2]  int64   sampled token pairs for edge prediction
        edge_labels:    [n_pairs]     int64   1 if edge exists, 0 otherwise
    """

    def __init__(
        self,
        npz_dir: str,
        bucket_dir: str,
        tokenizer,
        max_len: int = 2048,
        mlm_prob: float = 0.15,
        n_edge_pairs: int = 128,  # Increased for higher signal density
        shard_pattern: str = "v*_shard_*.npz",
        split: str = "train",
        val_frac: float = 0.05,
        test_frac: float = 0.05,
        seed: int = 42,
        objective_mode: str = "edge",  # kept for backward compat, ignored
        graph_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.n_edge_pairs = n_edge_pairs
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.graph_dir = graph_dir or npz_dir.replace("cpg_features", "metp_edges")

        # ── Discover all shards ──────────────────────────────────────
        shard_files = sorted(
            glob.glob(os.path.join(npz_dir, shard_pattern))
        )
        if not shard_files:
            raise FileNotFoundError(
                f"No shards matching {shard_pattern} in {npz_dir}"
            )

        # ── Build global index: (shard_path, idx_in_shard, contract_id) ──
        print(f"Indexing {len(shard_files)} shards...", flush=True)
        self.index = []
        for sf in shard_files:
            d = np.load(sf, allow_pickle=True)
            manifest = json.loads(d["manifest"][0])
            for entry in manifest:
                self.index.append({
                    "shard_path": sf,
                    "idx": entry["idx"],
                    "contract_id": entry["contract_id"],
                    "num_tokens": entry["num_tokens"],
                })
            d.close()

        print(f"Total contracts indexed: {len(self.index):,}", flush=True)

        # ── Build source code index from bucket files ────────────────
        print(f"Indexing source code from {bucket_dir}...", flush=True)
        self.source_index = {}  # contract_address → source_code
        bucket_files = sorted(
            glob.glob(os.path.join(bucket_dir, "*.jsonl"))
        )
        for bf in bucket_files:
            with open(bf) as f:
                for line in f:
                    rec = json.loads(line)
                    addr = rec.get("contract_address", "")
                    if addr:
                        self.source_index[addr] = rec["source_code"]
        print(
            f"Source code indexed: {len(self.source_index):,} contracts",
            flush=True,
        )

        # ── Filter to contracts that have both NPZ and source ────────
        valid = [
            e for e in self.index
            if e["contract_id"] in self.source_index
        ]
        dropped = len(self.index) - len(valid)
        if dropped:
            print(f"Dropped {dropped:,} contracts without source code")
        self.index = valid

        # ── Split into train/val/test ────────────────────────────────
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

        print(
            f"Split '{split}': {len(self.index):,} contracts", flush=True
        )

        # ── Shard LRU cache ──────────────────────────────────────────
        self._shard_cache = {}
        self._cache_order = []
        self._max_cached = 8  # keep at most 8 shards in RAM

        # ── Graph shard cache ────────────────────────────────────────
        self._graph_shard_cache = {}
        self._graph_cache_order = []
        self._graph_max_cached = 8

        # ── ERR shard cache (separate from feature shards) ───────────
        self._err_shard_cache = {}
        self._err_cache_order = []
        self._err_max_cached = 8

        # ── Compute node-type thresholds from corpus stats ───────────
        # Poisson (dim 0): mean=0, std=1
        # We'll use simple percentile-based thresholds
        # Poisson (dim 0): mean=0, std=1, range [-1.3, 5.2]
        self.poisson_source_thresh = 1.0   # > mean + 1 std → top ~16%
        self.poisson_sink_thresh = -0.8    # < mean - 0.8 std → bottom ~20%
        # Diffusion d_scale1 (dim 1): mean=-10.2, range [-19.4, -1.1]
        self.hub_thresh = -5.0             # top quartile connectivity
        # d_scale5 (dim 5): mean=-11.3, range [-19.4, -0.4]
        self.isolated_thresh = -17.0       # bottom ~15% connectivity

    def __len__(self):
        return len(self.index)

    def _get_shard(self, shard_path: str) -> dict:
        """Load shard with LRU caching."""
        if shard_path in self._shard_cache:
            return self._shard_cache[shard_path]

        # Evict oldest if cache full
        if len(self._shard_cache) >= self._max_cached:
            oldest = self._cache_order.pop(0)
            if oldest in self._shard_cache:
                del self._shard_cache[oldest]

        data = np.load(shard_path, allow_pickle=True)
        self._shard_cache[shard_path] = data
        self._cache_order.append(shard_path)
        return data

    def _get_graph_shard(self, feature_shard_path: str):
        """Load companion multiplex graph shard for a given feature shard."""
        shard_name = os.path.basename(feature_shard_path)
        graph_path = os.path.join(self.graph_dir, shard_name)

        if not os.path.exists(graph_path):
            return None

        if graph_path in self._graph_shard_cache:
            return self._graph_shard_cache[graph_path]

        if len(self._graph_shard_cache) >= self._graph_max_cached:
            oldest = self._graph_cache_order.pop(0)
            if oldest in self._graph_shard_cache:
                del self._graph_shard_cache[oldest]

        data = np.load(graph_path, allow_pickle=True)
        self._graph_shard_cache[graph_path] = data
        self._graph_cache_order.append(graph_path)
        return data

    def _get_err_shard(self, feature_shard_path: str):
        """Load companion ERR shard for a given feature shard."""
        # feature: v0.4_shard_0000.npz → err: err_v0.4_shard_0000.npz
        shard_dir = os.path.dirname(feature_shard_path)
        shard_name = os.path.basename(feature_shard_path)
        err_name = f"err_{shard_name}"
        err_path = os.path.join(shard_dir, err_name)

        if not os.path.exists(err_path):
            return None

        if err_path in self._err_shard_cache:
            return self._err_shard_cache[err_path]

        if len(self._err_shard_cache) >= self._err_max_cached:
            oldest = self._err_cache_order.pop(0)
            if oldest in self._err_shard_cache:
                del self._err_shard_cache[oldest]

        data = np.load(err_path, allow_pickle=True)
        self._err_shard_cache[err_path] = data
        self._err_cache_order.append(err_path)
        return data

    def _derive_node_types(self, node_features: np.ndarray) -> np.ndarray:
        """
        Derive node type labels from graph features.

        Returns int32 array of shape (N_nodes,) with values in [0, 4]:
            0: SOURCE     (high Poisson → upstream/entry)
            1: SINK       (low Poisson → downstream/exit)
            2: MIDDLE     (moderate Poisson)
            3: HUB        (high local connectivity)
            4: ISOLATED   (low global connectivity)

        Priority: SOURCE > SINK > HUB > ISOLATED > MIDDLE
        """
        N = node_features.shape[0]
        labels = np.full(N, 2, dtype=np.int32)  # default: MIDDLE

        poisson = node_features[:, 0]
        d_scale1 = node_features[:, 1]
        d_scale5 = node_features[:, 5]

        # Assign by priority (lower priority first, higher overwrites)
        labels[d_scale5 < self.isolated_thresh] = 4    # ISOLATED
        labels[d_scale1 > self.hub_thresh] = 3         # HUB
        labels[poisson < self.poisson_sink_thresh] = 1  # SINK
        labels[poisson > self.poisson_source_thresh] = 0  # SOURCE

        return labels

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        shard = self._get_shard(entry["shard_path"])
        ci = entry["idx"]  # index within shard

        # ── Load graph data ──────────────────────────────────────────
        node_features = shard[f"node_features_{ci}"]      # (N, 6)
        # [GUARD] Handle NaN/Inf in features
        if not np.isfinite(node_features).all():
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        token_to_node = shard[f"token_to_node_{ci}"]       # (T_raw,)
        token_features = shard[f"token_features_{ci}"]     # (T_raw, 6)

        # ── Get source code ──────────────────────────────────────────
        source_code = self.source_index[entry["contract_id"]]

        # ── Tokenize with ModernBERT ─────────────────────────────────
        encoding = self.tokenizer(
            source_code,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"][0]           # (max_len,)
        attention_mask = encoding["attention_mask"][0]  # (max_len,)
        offsets = encoding["offset_mapping"][0]         # (max_len, 2)

        seq_len = attention_mask.sum()  # actual length before padding

        # ── Map subword tokens → graph features ─────────────────────
        # Strategy: each subword token's char span overlaps with one
        # or more raw source tokens. We use the midpoint of the subword
        # span to find the corresponding raw token index, then look up
        # the graph feature via token_to_node.
        graph_feats = np.zeros((self.max_len, 6), dtype=np.float32)
        subword_to_node = np.full(self.max_len, -1, dtype=np.int64)

        # Strategy: Unmapped tokens get explicit ZERO (Null) coordinates.
        # This treats them as "structureless noise" (vacuum).
        # We iterate over subwords. If a subword maps to a valid source span
        # that maps to a graph node, we assign it. Otherwise, it stays 0.
        
        # Token-to-Node mapping source: `token_to_node` (raw token index -> node ID)
        T_raw = len(token_to_node)
        
        word_ids = encoding.word_ids(0) # For batch index 0
        
        for i in range(seq_len):
            word_idx = word_ids[i]
            if word_idx is None:
                continue # Special token
                
            if word_idx < T_raw:
                nid = token_to_node[word_idx]
                if nid >= 0: # Valid node
                     # In Null/Zero strategy, we copy ONLY if valid.
                     graph_feats[i] = node_features[nid]
                     subword_to_node[i] = nid
                # Else: nid -1 means unmapped -> leave as 0.0 (Null)

        # ── Node-Level Masking & TFR Targets ─────────────────────────
        # Strategy: Mask entirely by NODE, not by token.
        # This prevents leakage: if we mask Node X, we must mask ALL its tokens
        # so the model cannot peek at sibling tokens to guess the graph features.

        # 1. Group tokens by Node ID
        # node_id -> [list of seq_indices]
        node_to_tokens = {}
        content_nodes = []
        content_positions = []
        
        for i in range(seq_len):
            nid = subword_to_node[i]
            if nid >= 0:
                content_positions.append(i)
                if nid not in node_to_tokens:
                    node_to_tokens[nid] = []
                    content_nodes.append(nid)
                node_to_tokens[nid].append(i)

        # 2. Select Nodes to Mask (30% probability)
        n_mask_nodes = max(1, int(len(content_nodes) * self.mlm_prob))
        masked_node_ids = random.sample(
            content_nodes, 
            min(n_mask_nodes, len(content_nodes))
        )

        mlm_labels = np.full(self.max_len, -100, dtype=np.int64)
        masked_input = input_ids.copy()
        
        # Target for TFR: The 6D graph coordinates of the masked nodes
        # Use -100.0 (or similar sentinel) for unmasked, but for float 
        # we usually use a mask tensor. Here we'll zero-pad and use a separate mask.
        # But to keep API simple, we return dense targets and use mlm_labels != -100 to index.
        # Actually, let's return a specific `node_feature_targets` tensor.
        node_feature_targets = np.zeros((self.max_len, 6), dtype=np.float32)
        node_target_mask = np.zeros(self.max_len, dtype=np.float32) # 1.0 if tasked

        for nid in masked_node_ids:
            # Get the ground truth features for this node
            target_feat = node_features[nid] 
            
            # Normalize to [0, 1] for stable regression
            p5 = np.array(DEFAULT_FEAT_P5, dtype=np.float32)
            p95 = np.array(DEFAULT_FEAT_P95, dtype=np.float32)
            denom = p95 - p5
            denom[denom < 1e-6] = 1e-6
            normed_target = (np.clip(target_feat, p5, p95) - p5) / denom

            # Mask ALL tokens belonging to this node
            for pos in node_to_tokens[nid]:
                # MLM Target (Token ID) strategy
                mlm_labels[pos] = input_ids[pos]
                node_feature_targets[pos] = normed_target
                node_target_mask[pos] = 1.0
                
                # Standard BERT masking strategy
                r = random.random()
                if r < 0.8:
                    masked_input[pos] = self.mask_token_id
                elif r < 0.9:
                    masked_input[pos] = random.randint(0, self.vocab_size - 1)
                # else: keep original (10%)
                
                # else: keep original (10%)

        # ── Inter-Node Relation Objectives (always compute both) ─────
        # ── ERR: Effective Resistance Regression ──────────────────────
        # Load precomputed ERR pairs from companion shard.
        # Pairs are (node_A, node_B, R_eff) — map node IDs to token positions.
        err_pairs = np.zeros((self.n_edge_pairs, 2), dtype=np.int64)
        err_targets = np.zeros(self.n_edge_pairs, dtype=np.float32)

        err_shard = self._get_err_shard(entry["shard_path"])
        if err_shard is not None:
            err_key = f"err_pairs_{ci}"
            if err_key in err_shard:
                raw_err = err_shard[err_key]  # (K_raw, 3): [node_A, node_B, R_eff]

                if raw_err.shape[0] > 0:
                    # Filter NaN/Inf R_eff values
                    valid_mask = np.isfinite(raw_err[:, 2])
                    raw_err = raw_err[valid_mask]

                if raw_err.shape[0] > 0:
                    # Build reverse map: node_id → first token position
                    node_to_pos = {}
                    for pos in content_positions:
                        nid = subword_to_node[pos]
                        if nid >= 0 and nid not in node_to_pos:
                            node_to_pos[nid] = pos

                    # Map node pairs to token pairs
                    mapped_pairs = []
                    mapped_targets = []
                    for row in raw_err:
                        na, nb = int(row[0]), int(row[1])
                        r_eff = float(row[2])
                        if na in node_to_pos and nb in node_to_pos:
                            mapped_pairs.append([node_to_pos[na], node_to_pos[nb]])
                            # [GUARD] Clamp log-scaled targets to prevent explosion from corrupt data
                            val = np.log1p(abs(r_eff))
                            mapped_targets.append(np.clip(val, 0.0, 10.0))

                    if mapped_pairs:
                        mapped_pairs = np.array(mapped_pairs, dtype=np.int64)
                        mapped_targets = np.array(mapped_targets, dtype=np.float32)

                        K_avail = len(mapped_pairs)
                        if K_avail >= self.n_edge_pairs:
                            sel = random.sample(range(K_avail), self.n_edge_pairs)
                        else:
                            sel = [random.randint(0, K_avail - 1) for _ in range(self.n_edge_pairs)]

                        err_pairs = mapped_pairs[sel]
                        err_targets = mapped_targets[sel]

        # ── METP: Multiplex Edge-Type Prediction ───────────────────────
        edge_pairs = np.zeros((self.n_edge_pairs, 2), dtype=np.int64)
        edge_labels = np.full(self.n_edge_pairs, 3, dtype=np.int64)  # Default: 3 = NONE

        if len(content_positions) >= 2:
            graph_shard = self._get_graph_shard(entry["shard_path"])
            
            valid_true_pairs = []
            
            # node_to_pos maps node IDs to token positions
            node_to_pos = {}
            for pos in content_positions:
                nid = subword_to_node[pos]
                if nid >= 0 and nid not in node_to_pos: # Map first token of the node
                    node_to_pos[nid] = pos
            
            valid_nids_list = list(node_to_pos.keys())
            
            if graph_shard is not None and f"edges_{ci}" in graph_shard:
                edges_arr = graph_shard[f"edges_{ci}"] # shape (E, 3): [u, v, label]
                if len(edges_arr) > 0:
                    # Filter edges to only those where both u and v are in valid_nids
                    mask = np.isin(edges_arr[:, 0], valid_nids_list) & np.isin(edges_arr[:, 1], valid_nids_list)
                    valid_edges = edges_arr[mask]
                    
                    # Store as a list of (token_i, token_j, label)
                    for r in range(len(valid_edges)):
                        u, v, label = valid_edges[r]
                        valid_true_pairs.append((node_to_pos[u], node_to_pos[v], label))

            for ep in range(self.n_edge_pairs):
                label = 3  # 3 = NONE
                
                # 50% chance to sample a true positive edge (if available)
                if valid_true_pairs and random.random() < 0.5:
                    u_pos, v_pos, edge_label = random.choice(valid_true_pairs)
                    # Add noise to which specific token we pick if a node spans multiple tokens
                    i = random.choice(node_to_tokens.get(subword_to_node[u_pos], [u_pos]))
                    j = random.choice(node_to_tokens.get(subword_to_node[v_pos], [v_pos]))
                    label = edge_label
                else:
                    i, j = random.sample(content_positions, 2)
                    
                edge_pairs[ep] = [i, j]
                edge_labels[ep] = label

        return {
            "input_ids": torch.from_numpy(masked_input).long(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
            "graph_features": torch.from_numpy(graph_feats).float(),
            "mlm_labels": torch.from_numpy(mlm_labels).long(),
            "node_feature_targets": torch.from_numpy(
                node_feature_targets
            ).float(),
            "node_target_mask": torch.from_numpy(
                node_target_mask
            ).float(),
            "err_pairs": torch.from_numpy(err_pairs).long(),
            "err_targets": torch.from_numpy(err_targets).float(),
            "edge_pairs": torch.from_numpy(edge_pairs).long(),
            "edge_labels": torch.from_numpy(edge_labels).long(),
        }


def create_dataloaders(
    npz_dir: str,
    bucket_dir: str,
    tokenizer,
    max_len: int = 2048,
    mlm_prob: float = 0.30,
    n_edge_pairs: int = 128,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    objective_mode: str = "edge",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    common = dict(
        npz_dir=npz_dir,
        bucket_dir=bucket_dir,
        tokenizer=tokenizer,
        max_len=max_len,
        mlm_prob=mlm_prob,
        n_edge_pairs=n_edge_pairs,
        seed=seed,
        objective_mode=objective_mode,
    )

    train_ds = GraphRoPEDataset(split="train", **common)
    val_ds = GraphRoPEDataset(split="val", **common)
    test_ds = GraphRoPEDataset(split="test", **common)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl
