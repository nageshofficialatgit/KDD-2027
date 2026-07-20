#!/usr/bin/env python3
"""
Compute Effective Resistance (ERR) for Pre-Processed Contracts
--------------------------------------------------------------
Reads the original JSONL source files, rebuilds the transition matrix P
for each contract, samples K node pairs via STRATIFIED BFS SAMPLING,
and solves (I-P)f = e_A - e_B to compute pairwise effective resistance.

Stratified sampling:
  1. Run BFS from random anchors to compute hop distances
  2. Bucket all pairs by hop distance (1-hop, 2-hop, ...)
  3. Sample equally from each stratum → unbiased distance coverage

Outputs companion `.npz` shard files alongside the existing feature shards.

Usage:
    python compute_err.py \
        --jsonl /path/to/v0.5.jsonl \
        --output-dir /path/to/cpg_features \
        --bucket-name v0.5 \
        --pairs-per-contract 128 \
        --workers 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import bicgstab, lgmres, LinearOperator
from tqdm import tqdm

# --- Path setup (same as preprocess_contract.py) ---
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from graph_builder import build_cpg_from_source, IRNode
from random_walk import build_transition_matrix
from poisson_solver import build_pressure_vector
from combine import (
    compute_channel_stats, compute_lambdas, compute_lambdas_degree,
    compute_global_stats, mix_channels,
)
from solc_resolver import (
    resolve_binary, preflight_install, get_all_installed,
    detect_pragma_version, version_to_bucket,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERR Solver
# ---------------------------------------------------------------------------

def solve_err_pair(
    P: csr_matrix,
    node_a: int,
    node_b: int,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> float:
    """
    Compute effective resistance R_eff(A, B) by solving:
        (I - P + αI) f = e_A - e_B
    and returning f[A] - f[B].

    Uses warm-start BiCGSTAB with Jacobi preconditioning.
    Falls back to LGMRES on failure.

    Returns R_eff (float). Returns NaN on total failure.
    """
    N = P.shape[0]
    alpha = 1e-6  # regularization (same as poisson_solver.py)

    # Build RHS: e_A - e_B (unit vectors)
    b = np.zeros(N, dtype=np.float64)
    b[node_a] = 1.0
    b[node_b] = -1.0

    # Build system matrix: A_sys = I - P + αI
    A_sys = eye(N, format='csr') - P + alpha * eye(N, format='csr')

    # Jacobi preconditioner
    diag_vals = np.asarray(A_sys.diagonal()).flatten()
    safe_diag = np.where(np.abs(diag_vals) > 1e-12, diag_vals, 1.0)
    M_inv_diag = 1.0 / safe_diag
    M_inv = LinearOperator(
        shape=(N, N),
        matvec=lambda x: M_inv_diag * x,
    )

    # Attempt 1: BiCGSTAB
    try:
        f, info = bicgstab(A_sys, b, M=M_inv, maxiter=max_iter, atol=tol)
        if info == 0:
            return float(f[node_a] - f[node_b])
    except Exception:
        pass

    # Attempt 2: LGMRES fallback
    try:
        f, info = lgmres(A_sys, b, maxiter=max_iter, atol=tol)
        if info == 0:
            return float(f[node_a] - f[node_b])
    except Exception:
        pass

    return float('nan')


def _rw_stratified_sample_pairs(
    P: csr_matrix,
    num_nodes: int,
    K: int,
    rng: np.random.RandomState,
    max_walk_len: int = 20,
) -> list[tuple[int, int]]:
    """
    Stratified pair sampling via random walks on the transition matrix P.

    1. Start random walks from random anchor nodes using P
    2. At each walk step t, record (anchor, visited_node) as a pair
    3. Walk step t = proxy for graph distance → natural stratification
    4. Sample equally from each stratum (walk distance bucket)

    This is aligned with the same P used for ERR solving — no mismatch
    between BFS distances and the actual resistance metric.
    """
    # Choose anchors: ~20% of nodes, minimum 4
    n_anchors = min(num_nodes, max(4, num_nodes // 5))
    anchor_nodes = rng.choice(num_nodes, size=n_anchors, replace=False)

    # walks_per_anchor: ensure enough coverage
    walks_per_anchor = max(2, K // (n_anchors * max_walk_len) + 1)

    # Collect pairs by walk distance (step count)
    buckets = defaultdict(set)  # step -> set of (i, j) with i < j

    for anchor in anchor_nodes:
        for _ in range(walks_per_anchor):
            current = anchor
            for step in range(1, max_walk_len + 1):
                # Sample next node from P[current]
                row = P[current]
                if row.nnz == 0:
                    break  # dead-end node
                probs = np.asarray(row.todense()).flatten()
                probs = np.maximum(probs, 0)  # safety
                p_sum = probs.sum()
                if p_sum < 1e-12:
                    break
                probs /= p_sum
                current = rng.choice(num_nodes, p=probs)

                if current != anchor:
                    a, b = min(anchor, current), max(anchor, current)
                    buckets[step].add((a, b))

    # Sort strata by walk distance
    sorted_steps = sorted(buckets.keys())
    if not sorted_steps:
        # Fallback: uniform random if walks found nothing
        pairs = []
        for _ in range(min(K, num_nodes * (num_nodes - 1) // 2)):
            a, b = rng.choice(num_nodes, size=2, replace=False)
            pairs.append((min(a, b), max(a, b)))
        return pairs

    n_strata = len(sorted_steps)
    per_stratum = max(1, K // n_strata)
    remainder = K - per_stratum * n_strata

    selected = []
    for i, step in enumerate(sorted_steps):
        pool = list(buckets[step])
        # Give extra samples to shorter walks (more informative locally)
        quota = per_stratum + (1 if i < remainder else 0)
        if len(pool) <= quota:
            selected.extend(pool)
        else:
            idx = rng.choice(len(pool), size=quota, replace=False)
            selected.extend(pool[j] for j in idx)

    # If we still need more, fill from all discovered pairs
    if len(selected) < K:
        all_discovered = set()
        for pairs_set in buckets.values():
            all_discovered.update(pairs_set)
        remaining = list(all_discovered - set(selected))
        if remaining:
            need = K - len(selected)
            idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
            selected.extend(remaining[j] for j in idx)

    # Deduplicate and cap at K
    seen = set()
    final = []
    for pair in selected:
        if pair not in seen:
            seen.add(pair)
            final.append(pair)
        if len(final) >= K:
            break

    return final


def compute_err_for_contract(
    P: csr_matrix,
    A_combined: csr_matrix,
    num_nodes: int,
    K: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample K node pairs via STRATIFIED RANDOM WALKS on P and compute
    effective resistance.

    Strategy:
    - Random walks on P_total to discover pairs at various walk distances
    - Walk step = proxy for graph distance → natural stratification
    - Aligned with same P used for ERR solving (no BFS/adjacency mismatch)
    - Build A_sys ONCE (same for all pairs), only change RHS vector b

    Returns: float32 array of shape (K, 3):
        [:, 0] = node_A_id
        [:, 1] = node_B_id
        [:, 2] = R_eff value

    For small graphs (N < 4), returns empty array.
    """
    if num_nodes < 4:
        return np.zeros((0, 3), dtype=np.float32)

    rng = np.random.RandomState(seed)

    max_possible = num_nodes * (num_nodes - 1) // 2
    actual_K = min(K, max_possible)
    if actual_K == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Stratified pair sampling via random walks on P
    pairs = _rw_stratified_sample_pairs(P, num_nodes, actual_K, rng)
    actual_K = len(pairs)

    # Pre-build system matrix ONCE
    N = P.shape[0]
    alpha = 1e-6
    A_sys = eye(N, format='csr') - P + alpha * eye(N, format='csr')

    # Jacobi preconditioner
    diag_vals = np.asarray(A_sys.diagonal()).flatten()
    safe_diag = np.where(np.abs(diag_vals) > 1e-12, diag_vals, 1.0)
    M_inv_diag = 1.0 / safe_diag
    M_inv = LinearOperator(
        shape=(N, N),
        matvec=lambda x: M_inv_diag * x,
    )

    results = np.zeros((actual_K, 3), dtype=np.float32)
    for k_idx, (node_a, node_b) in enumerate(pairs):
        b = np.zeros(N, dtype=np.float64)
        b[node_a] = 1.0
        b[node_b] = -1.0

        r_eff = float('nan')
        try:
            f, info = bicgstab(A_sys, b, M=M_inv, maxiter=300, atol=1e-5)
            if info == 0:
                r_eff = float(f[node_a] - f[node_b])
            else:
                f, info = lgmres(A_sys, b, maxiter=300, atol=1e-5)
                if info == 0:
                    r_eff = float(f[node_a] - f[node_b])
        except Exception:
            pass

        results[k_idx, 0] = node_a
        results[k_idx, 1] = node_b
        results[k_idx, 2] = r_eff

    return results


# ---------------------------------------------------------------------------
# Single-Contract ERR Pipeline (minimal — only builds P, computes ERR)
# ---------------------------------------------------------------------------

def process_err_single(
    source_code: str,
    compiler_version: Optional[str] = None,
    solc_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    global_stats: Optional[dict] = None,
    pairs_per_contract: int = 128,
    contract_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Lightweight pipeline: build graph → transition matrix → ERR pairs.

    Skips: tokenization, feature assembly, node metadata.
    Only computes what's needed for ERR.
    """
    if not source_code or not source_code.strip():
        return None

    # Build CPG
    cpg = build_cpg_from_source(source_code, solc_path=solc_path, tmp_dir=tmp_dir)
    if cpg is None:
        return None

    nodes = cpg.nodes
    N = len(nodes)
    if N < 4:
        return None

    A_cfg, A_dfg, A_call = cpg.A_cfg, cpg.A_dfg, cpg.A_call

    # Quality gate
    total_edges = A_cfg.nnz + A_dfg.nnz + A_call.nnz
    if total_edges == 0:
        return None

    # Per-channel transition matrices
    A_channels = {"cfg": A_cfg, "dfg": A_dfg, "call": A_call}
    P_channels = {}
    for ch_name, A_ch in A_channels.items():
        P_channels[ch_name] = build_transition_matrix(A_ch)

    # Pressure vector for lambda computation
    b = build_pressure_vector(nodes)

    # Per-channel stats for lambda
    H_k = {}
    R_k = {}
    for ch_name, P_ch in P_channels.items():
        h, r, _ = compute_channel_stats(P_ch, b)
        H_k[ch_name] = h
        R_k[ch_name] = r

    # Compute lambda_k
    if global_stats is not None:
        lambda_k = compute_lambdas(H_k, R_k, global_stats)
    else:
        lambda_k = compute_lambdas_degree(A_channels)

    # Mix channels → P_total
    A_total = mix_channels(A_channels, lambda_k)
    P_total = build_transition_matrix(A_total)

    # Compute ERR pairs (stratified BFS sampling)
    # Use contract_id hash as seed for reproducibility
    seed = int(hashlib.md5((contract_id or "").encode()).hexdigest()[:8], 16) % (2**31)
    err_pairs = compute_err_for_contract(
        P_total, A_total, N, K=pairs_per_contract, seed=seed
    )

    return {
        "err_pairs": err_pairs,     # (K, 3): [node_a, node_b, R_eff]
        "num_nodes": N,
    }


# ---------------------------------------------------------------------------
# Worker Function
# ---------------------------------------------------------------------------

def _err_worker(args: tuple) -> Optional[tuple[str, dict]]:
    """Process one contract for ERR computation."""
    line_json = args[0]
    solc_map_json = args[1]
    tmp_dir = args[2]
    global_stats_json = args[3] if len(args) > 3 else None
    pairs_per_contract = args[4] if len(args) > 4 else 128

    contract_id = "unknown"
    try:
        item = json.loads(line_json)
        source = item.get("source_code", "")
        version = item.get("compiler_version", "")
        contract_id = item.get("contract_address") or item.get("contract_name") or str(uuid.uuid4())

        solc_map = json.loads(solc_map_json)
        solc_path = solc_map.get(version)

        if not solc_path:
            return (contract_id, {"error": f"no_solc:{version}"})

        global_stats = json.loads(global_stats_json) if global_stats_json else None
        result = process_err_single(
            source, version, solc_path=solc_path, tmp_dir=tmp_dir,
            global_stats=global_stats,
            pairs_per_contract=pairs_per_contract,
            contract_id=contract_id,
        )
        if result is None:
            return (contract_id, {"error": "cpg_build_failed"})

        return (contract_id, result)
    except Exception as e:
        return (contract_id, {"error": str(e)[:120]})


# ---------------------------------------------------------------------------
# Main Processing Loop
# ---------------------------------------------------------------------------

SHARD_SIZE = 1000  # Match existing shard size

def process_err_bucket(
    jsonl_path: str,
    solc_map: dict[str, str],
    output_dir: str,
    global_stats_path: Optional[str] = None,
    workers: int = 24,
    pairs_per_contract: int = 128,
    resume_from: int = 0,
):
    """
    Process all contracts in a JSONL file for ERR computation.

    Saves companion .npz shards with prefix 'err_'.
    Each shard contains:
        err_pairs_{i}: float32 (K, 3) — [node_a, node_b, R_eff]
    """
    os.makedirs(output_dir, exist_ok=True)
    bucket_name = Path(jsonl_path).stem

    # Load global stats (for λ_k computation)
    global_stats = None
    if global_stats_path and os.path.exists(global_stats_path):
        with open(global_stats_path) as f:
            global_stats = json.load(f)
        print(f"  Loaded global stats from {global_stats_path}", flush=True)
    else:
        print("  WARNING: No global stats — using degree-based λ_k fallback", flush=True)

    # Read JSONL lines
    print(f"  Loading {jsonl_path}...", flush=True)
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    total = len(lines)
    print(f"  Loaded {total:,} contracts", flush=True)

    if total == 0:
        return

    # Skip already-processed contracts (resume support)
    start_idx = resume_from * SHARD_SIZE
    if start_idx >= total:
        print(f"  Already completed (resume_from={resume_from}, total={total})", flush=True)
        return

    # Prepare worker args
    tmp_dir = os.path.join(output_dir, f".tmp_err_{bucket_name}")
    os.makedirs(tmp_dir, exist_ok=True)
    solc_map_json = json.dumps(solc_map)
    global_stats_json = json.dumps(global_stats) if global_stats else None

    # Process in shard-sized chunks for incremental saving
    shard_idx = resume_from
    processed = 0
    total_success = 0
    total_failed = 0

    for chunk_start in range(start_idx, total, SHARD_SIZE):
        chunk_end = min(chunk_start + SHARD_SIZE, total)
        chunk_lines = lines[chunk_start:chunk_end]

        args_list = [
            (line, solc_map_json, tmp_dir, global_stats_json, pairs_per_contract)
            for line in chunk_lines
        ]

        shard_data = {}
        shard_success = 0
        shard_failed = 0
        ci = 0  # contract index within this shard

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results_iter = executor.map(_err_worker, args_list, chunksize=32)
            desc = f"{bucket_name} ERR shard {shard_idx}"
            pbar = tqdm(
                results_iter, total=len(chunk_lines), desc=desc,
                unit="c", miniters=1, mininterval=2.0, ncols=90,
            )

            for result in pbar:
                if result is None or "error" in result[1]:
                    shard_failed += 1
                    # Still save empty ERR for this contract to keep alignment
                    shard_data[f"err_pairs_{ci}"] = np.zeros((0, 3), dtype=np.float32)
                else:
                    contract_id, data = result
                    shard_data[f"err_pairs_{ci}"] = data["err_pairs"]
                    shard_success += 1

                ci += 1
                pbar.set_postfix(ok=shard_success, fail=shard_failed)

        # Save shard
        shard_name = f"err_{bucket_name}_shard_{shard_idx:04d}.npz"
        shard_path = os.path.join(output_dir, shard_name)
        np.savez(shard_path, **shard_data)

        total_success += shard_success
        total_failed += shard_failed

        print(
            f"  ✓ Saved {shard_name} | "
            f"ok={shard_success} fail={shard_failed} | "
            f"cumulative: {total_success}/{total_success + total_failed}",
            flush=True,
        )
        shard_idx += 1

        # Checkpoint: write resume marker
        checkpoint_path = os.path.join(output_dir, f"err_{bucket_name}_checkpoint.json")
        with open(checkpoint_path, 'w') as f:
            json.dump({
                "last_completed_shard": shard_idx - 1,
                "total_success": total_success,
                "total_failed": total_failed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    print(
        f"\n  Done: {total_success} success, {total_failed} failed "
        f"across {shard_idx - resume_from} shards",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute ERR pairs for contracts")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL bucket file")
    parser.add_argument("--output-dir", required=True, help="Output directory for ERR shards")
    parser.add_argument("--global-stats", default=None, help="Path to global_stats.json")
    parser.add_argument("--bucket-name", default=None, help="Override bucket name")
    parser.add_argument("--pairs-per-contract", type=int, default=128, help="K pairs per contract (stratified BFS sampling)")
    parser.add_argument("--workers", type=int, default=24, help="Number of parallel workers")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from shard index")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Auto-detect global stats if not provided
    if args.global_stats is None:
        bucket = args.bucket_name or Path(args.jsonl).stem
        candidate = os.path.join(args.output_dir, f"{bucket}_global_stats.json")
        if os.path.exists(candidate):
            args.global_stats = candidate
            print(f"  Auto-detected global stats: {candidate}", flush=True)

    # Build solc map
    solc_map = get_all_installed()
    print(f"  Available solc versions: {len(solc_map)}", flush=True)

    # Auto-detect resume point from checkpoint
    bucket = args.bucket_name or Path(args.jsonl).stem
    checkpoint_path = os.path.join(args.output_dir, f"err_{bucket}_checkpoint.json")
    if args.resume_from == 0 and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        resume_shard = ckpt.get("last_completed_shard", -1) + 1
        if resume_shard > 0:
            print(f"  ★ Auto-resuming from shard {resume_shard} (checkpoint found)", flush=True)
            args.resume_from = resume_shard

    process_err_bucket(
        jsonl_path=args.jsonl,
        solc_map=solc_map,
        output_dir=args.output_dir,
        global_stats_path=args.global_stats,
        workers=args.workers,
        pairs_per_contract=args.pairs_per_contract,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
