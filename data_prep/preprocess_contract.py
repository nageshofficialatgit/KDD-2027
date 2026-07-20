#!/usr/bin/env python3
"""
Preprocess Contract — End-to-End Pipeline + CLI
------------------------------------------------
Combines graph building, random walk, and Poisson solving into a single
pipeline.  Supports single-file and batch (multiprocessing) modes.

Usage:
    # Single contract
    python preprocess_contract.py contract.sol

    # Batch over cleaned buckets (multiprocessing)
    python preprocess_contract.py --batch Dataset_for_first/cleaned_buckets/ \
        --output-dir model/cpg_features/ --workers 48

Output per shard:  shard_XXXX.npz
"""

from __future__ import annotations

from scipy.sparse import coo_matrix, csr_matrix

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import re
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_builder import build_cpg, build_cpg_from_source, IRNode, CPGResult  # noqa: E402
from random_walk import build_transition_matrix, select_anchors, multi_scale_diffusion  # noqa: E402
from poisson_solver import build_pressure_vector, solve_poisson  # noqa: E402
from combine import (  # noqa: E402
    compute_channel_stats, compute_lambdas, compute_lambdas_degree,
    compute_global_stats, mix_channels,
)
from solc_resolver import (  # noqa: E402
    resolve_binary, preflight_install, get_all_installed,
    detect_pragma_version, version_to_bucket,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "model" / "cpg_features")
DEFAULT_GRAPH_DIR = str(PROJECT_ROOT / "model" / "cpg_graphs")
SHARD_SIZE = 1000  # contracts per shard


# ---------------------------------------------------------------------------
# Tokenizer (standalone, no legacy dependency)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    0x[a-fA-F0-9]+              |   # hex literals
    [A-Za-z_][A-Za-z0-9_]*      |   # identifiers / keywords
    \d+\.\d+  |  \d+            |   # numbers
    ==|!=|<=|>=|&&|\|\|         |   # multi-char ops
    ->|=>|\+\+|--|<<|>>|::     |
    [^\s]                           # single non-ws char
    """,
    re.VERBOSE,
)


def tokenize_source(code: str) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Heuristic Solidity tokenizer.
    Returns (tokens, spans) where each span is (start_char, end_char).
    """
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(code):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


# ---------------------------------------------------------------------------
# Token ↔ Node Mapping
# ---------------------------------------------------------------------------

def _assign_source_spans(
    nodes: list[IRNode],
    tokens: list[str],
    token_spans: list[tuple[int, int]],
    source_code: str,
) -> None:
    """
    Assign (start_token_idx, end_token_idx) spans to each IR node based
    on its source line information.  Modifies nodes in-place.
    """
    if not tokens or not source_code:
        return

    # Build a line→char-offset map
    line_offsets: list[int] = [0]
    for ch in source_code:
        if ch == '\n':
            line_offsets.append(len(line_offsets))  # placeholder
    # Real offsets
    line_offsets = [0]
    for i, ch in enumerate(source_code):
        if ch == '\n':
            line_offsets.append(i + 1)

    for node in nodes:
        if not node.source_lines:
            node.source_span = (-1, -1)
            continue

        # Find char range from source lines
        min_line = min(node.source_lines)
        max_line = max(node.source_lines)

        if min_line < 1 or min_line > len(line_offsets):
            node.source_span = (-1, -1)
            continue

        char_start = line_offsets[min_line - 1] if (min_line - 1) < len(line_offsets) else 0
        if max_line < len(line_offsets):
            char_end = line_offsets[max_line]  # start of next line
        else:
            char_end = len(source_code)

        # Map char range to token indices
        tok_start = -1
        tok_end = -1
        for i, (ts, te) in enumerate(token_spans):
            if te > char_start:
                tok_start = i
                break
        for i in range(len(token_spans) - 1, -1, -1):
            if token_spans[i][0] < char_end:
                tok_end = i
                break

        if tok_start >= 0 and tok_end >= tok_start:
            node.source_span = (tok_start, tok_end)
        else:
            node.source_span = (-1, -1)


def map_tokens_to_nodes(
    tokens: list[str],
    token_spans: list[tuple[int, int]],
    nodes: list[IRNode],
) -> np.ndarray:
    """
    Map each token to a node id.

    Rules:
    - If token inside some node's source_span → map directly.
    - If not inside any span → assign to nearest preceding node.
    - If before first node → assign to first entry node.

    Returns int32 array of shape (num_tokens,).
    """
    T = len(tokens)
    N = len(nodes)

    if T == 0 or N == 0:
        return np.zeros(T, dtype=np.int32)

    mapping = np.full(T, -1, dtype=np.int32)

    # Build sorted list of (tok_start, tok_end, node_id) for nodes with valid spans
    node_spans: list[tuple[int, int, int]] = []
    for node in nodes:
        s, e = node.source_span
        if s >= 0 and e >= 0:
            node_spans.append((s, e, node.id))

    # Sort by starting token index
    node_spans.sort(key=lambda x: x[0])

    # Direct assignment: for each token, check if it falls inside a node span
    # Use a sweep approach for efficiency
    span_idx = 0
    for tok_idx in range(T):
        # Advance span_idx to the first span that could contain tok_idx
        while span_idx < len(node_spans) and node_spans[span_idx][1] < tok_idx:
            span_idx += 1

        # Check overlapping spans
        assigned = False
        for si in range(max(0, span_idx - 1), len(node_spans)):
            s, e, nid = node_spans[si]
            if s > tok_idx:
                break
            if s <= tok_idx <= e:
                mapping[tok_idx] = nid
                assigned = True
                break

    # Fill unmapped tokens: nearest preceding node
    first_entry = 0
    for node in nodes:
        if node.is_entry:
            first_entry = node.id
            break

    last_assigned = first_entry
    for tok_idx in range(T):
        if mapping[tok_idx] >= 0:
            last_assigned = mapping[tok_idx]
        else:
            mapping[tok_idx] = last_assigned

    return mapping


# ---------------------------------------------------------------------------
# Feature Assembly
# ---------------------------------------------------------------------------

def assemble_features(
    poisson_f: np.ndarray,
    diffusion_g: np.ndarray,
) -> np.ndarray:
    """
    Compose structural feature vector S(i) = [f(i), g_1, g_2, g_4, g_8, g_16].
    Returns float32 array of shape (N, 6).
    """
    f_col = poisson_f.astype(np.float32).reshape(-1, 1)
    return np.hstack([f_col, diffusion_g]).astype(np.float32)


def _sparse_to_coo_arrays(mat: csr_matrix) -> dict[str, np.ndarray]:
    """Convert a sparse matrix to COO arrays for npz storage."""
    coo = coo_matrix(mat)
    return {
        "row": coo.row.astype(np.int32),
        "col": coo.col.astype(np.int32),
        "data": coo.data.astype(np.float32),
        "shape": np.array(coo.shape, dtype=np.int32),
    }


def _nodes_to_metadata(nodes: list[IRNode]) -> dict[str, np.ndarray]:
    """Serialize node metadata to numpy arrays."""
    N = len(nodes)
    opcodes = np.array([n.opcode for n in nodes], dtype=object)
    func_names = np.array([n.function_name for n in nodes], dtype=object)
    flags = np.zeros((N, 4), dtype=np.int8)  # [is_entry, is_exit, is_ext_call, is_state_write]
    for i, n in enumerate(nodes):
        flags[i, 0] = int(n.is_entry)
        flags[i, 1] = int(n.is_exit)
        flags[i, 2] = int(n.is_external_call)
        flags[i, 3] = int(n.is_state_write)
    return {
        "opcodes": opcodes,
        "func_names": func_names,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Single-Contract Pipeline
# ---------------------------------------------------------------------------

def process_single(
    source_code: str,
    compiler_version: Optional[str] = None,
    solc_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    global_stats: Optional[dict] = None,
    stats_only: bool = False,
    cpg_cache_dir: Optional[str] = None,
    contract_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Full pipeline for one contract.

    Per-channel processing:
        1. Build P_cfg, P_dfg, P_call (each with teleport)
        2. Compute per-channel H_k and R_k
        3. Compute λ_k via combine module:
           - If global_stats provided: robust norm + harmonic mean
           - Else: degree-based fallback
        4. Mix A_total = Σ λ_k A_k, build P_total
        5. Compute features on P_total

    Parameters
    ----------
    global_stats : dict, optional
        Corpus-wide median/IQR stats for robust λ_k normalization.
    stats_only : bool
        If True, return only H_k/R_k (for two-pass batch stats collection).
    cpg_cache_dir : str, optional
        Directory for CPG pickle cache.  Pass 1 writes, Pass 2 reads.
    contract_id : str, optional
        Used as cache key (hashed with source for safety).

    Returns dict with:
        node_features:  float32[N, 6]
        token_to_node:  int32[T]
        token_features: float32[T, 6]
        graph_cfg/dfg/call: COO triplet dicts
        node_meta:       metadata dict
        num_nodes: int
        num_tokens: int
        lambda_k: dict           # mixing weights
        diagnostics: dict        # per-contract diagnostics
    or None on failure.
    """
    if not source_code or not source_code.strip():
        return None

    # --- Cache key ---
    cache_path = None
    if cpg_cache_dir and contract_id:
        cache_key = hashlib.md5((contract_id + source_code[:200]).encode()).hexdigest()
        cache_path = os.path.join(cpg_cache_dir, f"{cache_key}.pkl")

    # --- Try loading from cache (Pass 2) ---
    cpg_loaded = False
    if cache_path and not stats_only and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as cf:
                cached = pickle.load(cf)
            A_cfg = cached['A_cfg']
            A_dfg = cached['A_dfg']
            A_call = cached['A_call']
            nodes = cached['nodes']
            source_code = cached['source_code']
            N = len(nodes)
            cpg_loaded = True
        except Exception:
            cpg_loaded = False  # fall through to rebuild

    if not cpg_loaded:
        # Resolve solc binary
        if solc_path is None and compiler_version:
            solc_path = resolve_binary(compiler_version)

        # Build CPG
        cpg = build_cpg_from_source(source_code, solc_path=solc_path, tmp_dir=tmp_dir)
        if cpg is None:
            return None

        nodes = cpg.nodes
        N = len(nodes)
        if N == 0:
            return None

        A_cfg, A_dfg, A_call = cpg.A_cfg, cpg.A_dfg, cpg.A_call

        # --- Save to cache (Pass 1) ---
        if cache_path and stats_only:
            try:
                with open(cache_path, 'wb') as cf:
                    pickle.dump({
                        'A_cfg': A_cfg, 'A_dfg': A_dfg, 'A_call': A_call,
                        'nodes': nodes, 'source_code': source_code,
                    }, cf, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass  # non-fatal — Pass 2 will just rebuild

    # --- Minimum quality gate: skip degenerate graphs ---
    total_edges = A_cfg.nnz + A_dfg.nnz + A_call.nnz
    if N < 3 or total_edges == 0:
        return None

    # --- Per-channel transition matrices ---
    A_channels = {"cfg": A_cfg, "dfg": A_dfg, "call": A_call}
    P_channels = {}
    for ch_name, A_ch in A_channels.items():
        P_channels[ch_name] = build_transition_matrix(A_ch)

    # --- Pressure vector (shared across channels) ---
    b = build_pressure_vector(nodes)

    # --- Per-channel entropy (H_k) and resistance (R_k) ---
    H_k: dict[str, float] = {}
    R_k: dict[str, float] = {}
    channel_status: dict[str, str] = {}
    for ch_name, P_ch in P_channels.items():
        h, r, st = compute_channel_stats(P_ch, b)
        H_k[ch_name] = h
        R_k[ch_name] = r
        channel_status[ch_name] = st

    # --- Stats-only mode (two-pass batch, pass 1) ---
    if stats_only:
        return {
            "H_k": H_k,
            "R_k": R_k,
            "channel_status": channel_status,
            "num_nodes": N,
        }

    # --- Compute λ_k ---
    if global_stats is not None:
        lambda_k = compute_lambdas(H_k, R_k, global_stats)
    else:
        # Single-contract fallback: degree-based normalization
        lambda_k = compute_lambdas_degree(A_channels)

    # --- Mix channels → A_total → P_total ---
    A_total = mix_channels(A_channels, lambda_k)
    P_total = build_transition_matrix(A_total)

    # --- Tokenize source ---
    tokens, token_spans = tokenize_source(source_code)

    # --- Assign source spans to nodes ---
    _assign_source_spans(nodes, tokens, token_spans, source_code)

    # --- Random walk: anchors + diffusion on P_total ---
    v0 = select_anchors(nodes)
    diffusion_g = multi_scale_diffusion(v0, P_total)

    # --- Poisson potential on P_total ---
    f_final, final_res, final_iters, final_status = solve_poisson(P_total, b)

    # --- Assemble features ---
    node_features = assemble_features(f_final, diffusion_g)

    # --- Token mapping ---
    token_to_node = map_tokens_to_nodes(tokens, token_spans, nodes)
    token_features = node_features[token_to_node]

    # --- Serialize graph structure ---
    graph_cfg = _sparse_to_coo_arrays(A_cfg)
    graph_dfg = _sparse_to_coo_arrays(A_dfg)
    graph_call = _sparse_to_coo_arrays(A_call)
    node_meta = _nodes_to_metadata(nodes)

    # --- Build diagnostics ---
    diagnostics = {
        "num_nodes": N,
        "num_tokens": len(tokens),
        "lambda_k": lambda_k,
        "channel_H_k": H_k,
        "channel_R_k": R_k,
        "channel_status": channel_status,
        "final_solver": {
            "residual": float(final_res),
            "iterations": int(final_iters),
            "status": final_status,
        },
        "edge_counts": {
            "cfg": int(A_cfg.nnz),
            "dfg": int(A_dfg.nnz),
            "call": int(A_call.nnz),
        },
    }

    return {
        "node_features": node_features,
        "token_to_node": token_to_node,
        "token_features": token_features,
        "graph_cfg": graph_cfg,
        "graph_dfg": graph_dfg,
        "graph_call": graph_call,
        "node_meta": node_meta,
        "num_nodes": N,
        "num_tokens": len(tokens),
        "lambda_k": lambda_k,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Worker Function (must be picklable — standalone)
# ---------------------------------------------------------------------------

def _worker_fn(args: tuple) -> Optional[tuple[str, dict]]:
    """
    Process one contract line from JSONL.

    args tuple format:
        (line_json, solc_map_json, tmp_dir, global_stats_json, stats_only, cpg_cache_dir)

    Returns (contract_id, result_dict) on success,
    or (contract_id, {"error": msg}) on failure.
    """
    # Unpack — supports variable-length tuples for backward compat
    line_json = args[0]
    solc_map_json = args[1]
    tmp_dir = args[2]
    global_stats_json = args[3] if len(args) > 3 else None
    stats_only = args[4] if len(args) > 4 else False
    cpg_cache_dir = args[5] if len(args) > 5 else None

    contract_id = "unknown"
    try:
        item = json.loads(line_json)
        source = item.get("source_code", "")
        version = item.get("compiler_version")
        if not version and source:
            from solc_resolver import detect_pragma_version
            version = detect_pragma_version(source)
        if not version:
            version = ""
        contract_id = item.get("filename") or item.get("contract_address") or item.get("contract_name") or str(uuid.uuid4())

        solc_map = json.loads(solc_map_json)
        solc_path = solc_map.get(version)

        if not solc_path:
            return (contract_id, {"error": f"no_solc:{version}"})

        global_stats = json.loads(global_stats_json) if global_stats_json else None
        result = process_single(
            source, version, solc_path=solc_path, tmp_dir=tmp_dir,
            global_stats=global_stats, stats_only=stats_only,
            cpg_cache_dir=cpg_cache_dir, contract_id=contract_id,
        )
        if result is None:
            return (contract_id, {"error": f"cpg_build_failed:v={version}:p={solc_path}"})

        return (contract_id, result)
    except Exception as e:
        return (contract_id, {"error": str(e)[:80]})


# ---------------------------------------------------------------------------
# Batch Processing with Shard I/O
# ---------------------------------------------------------------------------

def process_bucket(
    jsonl_path: str,
    solc_map: dict[str, str],
    output_dir: str,
    graph_dir: Optional[str] = None,
    workers: int = 48,
    shard_size: int = SHARD_SIZE,
) -> dict[str, int]:
    """
    Process all contracts in a single JSONL bucket file using two-pass pipeline.

    Pass 1: Collect per-channel H_k, R_k stats (lightweight, stats_only mode).
    Pass 2: Compute features using corpus-wide global_stats for robust λ_k.

    Outputs feature .npz shards to output_dir.
    If graph_dir is set, outputs graph .npz shards there (matching shard names).

    Returns stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    bucket_name = Path(jsonl_path).stem  # e.g. "v0.8"

    # Read lines with progress
    file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    print(f"  Loading {jsonl_path} ({file_size_mb:.0f} MB)...", flush=True)
    lines: list[str] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
            if len(lines) % 50000 == 0 and len(lines) > 0:
                print(f"    ... loaded {len(lines):,} lines", flush=True)
    total = len(lines)
    print(f"  Loaded {total:,} contracts", flush=True)

    if total == 0:
        return {"total": 0, "success": 0, "failed": 0}

    # Prepare worker arguments
    tmp_dir = os.path.join(output_dir, f".tmp_{bucket_name}")
    os.makedirs(tmp_dir, exist_ok=True)
    solc_map_json = json.dumps(solc_map)

    # CPG cache directory (Pass 1 writes, Pass 2 reads)
    cpg_cache_dir = os.path.join(output_dir, f".cpg_cache_{bucket_name}")
    os.makedirs(cpg_cache_dir, exist_ok=True)

    # ===================================================================
    # PASS 1: Collect per-channel H_k, R_k stats
    # ===================================================================
    print(f"\n  --- Pass 1: Collecting channel stats (+ caching CPGs) ---", flush=True)
    args_pass1 = [(line, solc_map_json, tmp_dir, None, True, cpg_cache_dir) for line in lines]

    H_all: dict[str, list[float]] = {"cfg": [], "dfg": [], "call": []}
    R_all: dict[str, list[float]] = {"cfg": [], "dfg": [], "call": []}
    pass1_ok = 0
    pass1_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results_iter = executor.map(_worker_fn, args_pass1, chunksize=64)
        pbar = tqdm(
            results_iter, total=total, desc=f"{bucket_name} P1",
            unit="c", miniters=1, mininterval=1.0, ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        )
        pbar.set_postfix(ok=0, fail=0)

        for result in pbar:
            if result is None:
                pass1_fail += 1
            else:
                contract_id, data = result
                if "error" in data:
                    pass1_fail += 1
                elif "H_k" in data:
                    for ch in ("cfg", "dfg", "call"):
                        H_all[ch].append(data["H_k"].get(ch, 0.0))
                        R_all[ch].append(data["R_k"].get(ch, 0.0))
                    pass1_ok += 1
                else:
                    pass1_fail += 1
            pbar.set_postfix(ok=pass1_ok, fail=pass1_fail)

    del args_pass1

    print(f"  Pass 1 done: {pass1_ok} stats collected, {pass1_fail} failed", flush=True)

    # Compute global stats from collected H/R values
    if pass1_ok > 0:
        # Convert to numpy arrays
        H_all_np = {ch: np.array(vals) for ch, vals in H_all.items()}
        R_all_np = {ch: np.array(vals) for ch, vals in R_all.items()}
        global_stats = compute_global_stats(H_all_np, R_all_np)

        # Save global stats
        stats_path = os.path.join(output_dir, f"{bucket_name}_global_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(global_stats, f, indent=2)
        print(f"  Saved global stats: {stats_path}", flush=True)
        for ch in ("cfg", "dfg", "call"):
            print(f"    {ch}: med_H={global_stats['med_H'][ch]:.4f}, "
                  f"iqr_H={global_stats['iqr_H'][ch]:.4f}, "
                  f"med_L={global_stats['med_L'][ch]:.4f}, "
                  f"iqr_L={global_stats['iqr_L'][ch]:.4f}", flush=True)
    else:
        global_stats = None
        print("  WARNING: No stats collected, using degree-based λ_k fallback", flush=True)

    del H_all, R_all

    # ===================================================================
    # PASS 2: Process with global stats → features
    # ===================================================================
    print(f"\n  --- Pass 2: Computing features with global stats (using CPG cache) ---", flush=True)
    global_stats_json = json.dumps(global_stats) if global_stats else None
    args_pass2 = [(line, solc_map_json, tmp_dir, global_stats_json, False, cpg_cache_dir) for line in lines]
    del lines  # free memory

    # Process
    shard_idx = 0
    feat_data: dict[str, np.ndarray] = {}
    graph_data: dict[str, np.ndarray] = {}
    shard_manifest: list[dict] = []
    shard_diagnostics: list[dict] = []
    success = 0
    failed = 0
    contracts_in_shard = 0
    error_counts: dict[str, int] = {}

    # Diagnostics output directory
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    def _flush_shard():
        nonlocal shard_idx, feat_data, graph_data, shard_manifest, shard_diagnostics, contracts_in_shard
        if not feat_data:
            return
        manifest_arr = np.array([json.dumps(shard_manifest)], dtype=object)
        shard_name = f"{bucket_name}_shard_{shard_idx:04d}.npz"

        # Save feature shard
        feat_data["manifest"] = manifest_arr
        np.savez(os.path.join(output_dir, shard_name), **feat_data)

        # Save graph shard (separate file, same name)
        if graph_dir and graph_data:
            graph_data["manifest"] = manifest_arr
            np.savez(os.path.join(graph_dir, shard_name), **graph_data)

        # Save per-contract diagnostics JSONL
        diag_path = os.path.join(diag_dir, f"{bucket_name}_shard_{shard_idx:04d}_diagnostics.jsonl")
        with open(diag_path, 'w') as df:
            for diag in shard_diagnostics:
                df.write(json.dumps(diag) + '\n')

        print(f"    Saved shard {shard_idx} ({contracts_in_shard} contracts)", flush=True)
        shard_idx += 1
        feat_data = {}
        graph_data = {}
        shard_manifest = []
        shard_diagnostics = []
        contracts_in_shard = 0

    print(f"  Starting {total:,} tasks across {workers} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Use map with chunksize — streams results without upfront stall
        results_iter = executor.map(_worker_fn, args_pass2, chunksize=64)

        pbar = tqdm(
            results_iter,
            total=total,
            desc=f"{bucket_name} P2",
            unit="c",
            miniters=1,
            mininterval=1.0,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        )
        pbar.set_postfix(ok=0, fail=0)

        for result in pbar:
            if result is None:
                failed += 1
                error_counts["unknown"] = error_counts.get("unknown", 0) + 1
            else:
                contract_id, data = result
                if "error" in data:
                    failed += 1
                    err_key = data["error"].split(":")[0] if ":" in data["error"] else data["error"]
                    error_counts[err_key] = error_counts.get(err_key, 0) + 1
                else:
                    idx = contracts_in_shard
                    # Feature shard: only features
                    feat_data[f"node_features_{idx}"] = data["node_features"]
                    feat_data[f"token_to_node_{idx}"] = data["token_to_node"]
                    feat_data[f"token_features_{idx}"] = data["token_features"]
                    # Graph shard: adjacency + node metadata
                    if graph_dir:
                        for gname in ("graph_cfg", "graph_dfg", "graph_call"):
                            for k, v in data[gname].items():
                                graph_data[f"{gname}_{k}_{idx}"] = v
                        for k, v in data["node_meta"].items():
                            graph_data[f"node_{k}_{idx}"] = v
                    shard_manifest.append({
                        "idx": idx,
                        "contract_id": contract_id,
                        "num_nodes": data["num_nodes"],
                        "num_tokens": data["num_tokens"],
                        "lambda_k": data.get("lambda_k", {}),
                    })
                    # Collect diagnostics
                    diag_entry = data.get("diagnostics", {})
                    diag_entry["contract_id"] = contract_id
                    diag_entry["shard_idx"] = shard_idx
                    diag_entry["idx_in_shard"] = idx
                    shard_diagnostics.append(diag_entry)
                    contracts_in_shard += 1
                    success += 1

                    if contracts_in_shard >= shard_size:
                        _flush_shard()

            pbar.set_postfix(ok=success, fail=failed)

    # Final flush
    _flush_shard()

    # Print error breakdown
    if error_counts:
        print(f"  Error breakdown:")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {err}: {cnt:,}")

    # Cleanup tmp dir + CPG cache
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(cpg_cache_dir, ignore_errors=True)
    except Exception:
        pass

    return {"total": total, "success": success, "failed": failed}


# ---------------------------------------------------------------------------
# Single-File Pipeline
# ---------------------------------------------------------------------------

def preprocess_contract(
    sol_path: str,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    CLI entry point for a single .sol file.
    Reads, builds graph, computes features, saves .npz.
    Returns output path or None.
    """
    if not os.path.isfile(sol_path):
        print(f"Error: File not found: {sol_path}", file=sys.stderr)
        return None

    with open(sol_path, 'r', encoding='utf-8', errors='ignore') as f:
        source = f.read()

    version = detect_pragma_version(source)
    solc_path = resolve_binary(version) if version else None

    result = process_single(source, version, solc_path=solc_path)
    if result is None:
        print(f"Error: Failed to process {sol_path}", file=sys.stderr)
        return None

    if output_path is None:
        output_path = os.path.splitext(sol_path)[0] + "_features.npz"

    # Save features
    np.savez(
        output_path,
        node_features=result["node_features"],
        token_to_node=result["token_to_node"],
        token_features=result["token_features"],
    )
    # Save graph separately
    graph_path = os.path.splitext(sol_path)[0] + "_graph.npz"
    graph_dict: dict[str, np.ndarray] = {}
    for gname in ("graph_cfg", "graph_dfg", "graph_call"):
        for k, v in result[gname].items():
            graph_dict[f"{gname}_{k}"] = v
    for k, v in result["node_meta"].items():
        graph_dict[f"node_{k}"] = v
    np.savez(graph_path, **graph_dict)

    # Save per-contract diagnostics JSON
    diag_path = os.path.splitext(sol_path)[0] + "_diagnostics.json"
    diag = result.get("diagnostics", {})
    diag["source_file"] = os.path.basename(sol_path)
    with open(diag_path, 'w') as df:
        json.dump(diag, df, indent=2)

    print(f"Saved features:    {output_path}")
    print(f"Saved graph:       {graph_path}")
    print(f"Saved diagnostics: {diag_path}")
    print(f"  Nodes:  {result['num_nodes']}")
    print(f"  Tokens: {result['num_tokens']}")
    print(f"  Graph edges: CFG={len(result['graph_cfg']['data'])}, DFG={len(result['graph_dfg']['data'])}, Call={len(result['graph_call']['data'])}")
    print(f"  Feature dims: {result['node_features'].shape[1]}")
    print(f"  λ_k: {result.get('lambda_k', {})}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse CPG Preprocessing Pipeline"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a .sol file (single-file mode)",
    )
    parser.add_argument(
        "--batch",
        help="Directory of cleaned bucket JSONL files (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .npz shards",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.cpu_count() * 0.75)),
        help="Number of parallel workers (default: 0.75 × vCPU)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help="Contracts per output shard (default: 1000)",
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help="Output directory for graph .npz shards (default: model/cpg_graphs/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip buckets that already have output shards (resume after crash)",
    )
    args = parser.parse_args()

    if args.batch:
        # Batch mode: process all JSONL bucket files
        bucket_dir = args.batch
        if not os.path.isdir(bucket_dir):
            print(f"Error: Bucket directory not found: {bucket_dir}", file=sys.stderr)
            sys.exit(1)

        # Preflight: get all installed solc binaries
        print("Resolving solc binaries...")
        solc_map = get_all_installed()
        print(f"  Found {len(solc_map)} solc versions")

        # Find JSONL files
        jsonl_files = sorted([
            os.path.join(bucket_dir, f)
            for f in os.listdir(bucket_dir)
            if f.endswith('.jsonl')
        ])
        print(f"  Found {len(jsonl_files)} bucket files")

        os.makedirs(args.output_dir, exist_ok=True)

        # Resume: detect already-processed buckets
        if args.resume:
            import glob
            existing_shards = glob.glob(os.path.join(args.output_dir, "*_shard_*.npz"))
            done_buckets = set()
            for s in existing_shards:
                # e.g. "v0.6_shard_0003.npz" → "v0.6"
                basename = os.path.basename(s)
                bucket_prefix = basename.rsplit("_shard_", 1)[0]
                done_buckets.add(bucket_prefix)
            if done_buckets:
                print(f"  Resume mode: skipping {len(done_buckets)} already-processed buckets: {sorted(done_buckets)}")
            jsonl_files = [
                f for f in jsonl_files
                if Path(f).stem not in done_buckets
            ]
            print(f"  Remaining buckets to process: {len(jsonl_files)}")

        for jsonl_path in jsonl_files:
            print(f"\n--- Processing: {os.path.basename(jsonl_path)} ---")
            stats = process_bucket(
                jsonl_path,
                solc_map,
                args.output_dir,
                graph_dir=args.graph_dir,
                workers=args.workers,
                shard_size=args.shard_size,
            )
            print(f"  Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")

    elif args.input:
        # Single-file mode
        preprocess_contract(args.input)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
