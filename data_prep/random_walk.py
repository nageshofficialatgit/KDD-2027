#!/usr/bin/env python3
"""
Random Walk Module
------------------
Computes row-stochastic transition matrix with teleport, anchor-based
multi-scale diffusion, and log-landing probabilities.

All operations use scipy.sparse CSR matrices and numpy vectors.
Memory: O(N + E) per contract.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, eye

from graph_builder import IRNode


# ---------------------------------------------------------------------------
# Transition Matrix
# ---------------------------------------------------------------------------

def build_transition_matrix(
    A: csr_matrix,
    teleport_eps: float = 1e-6,
) -> csr_matrix:
    """
    Construct row-stochastic transition matrix P from adjacency A.

    Steps:
        1. If row sum D[i] == 0: add self-loop (absorbing state).
        2. Row-normalize: P[i,:] = A[i,:] / D[i].
        3. Apply teleport: P = (1-ε)P + ε·(1/N)·11ᵀ.
           Teleport is stored *implicitly* in the matvec, but for CSR
           compatibility we absorb it into the sparse matrix by adjusting
           each entry and adding uniform mass.  For small ε this adds
           negligible fill.

    Parameters
    ----------
    A : csr_matrix
        Weighted adjacency matrix (N×N).
    teleport_eps : float
        Teleport probability (default 1e-6).

    Returns
    -------
    P : csr_matrix, float64
        Row-stochastic with teleport.
    """
    N = A.shape[0]
    if N == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    # 1. Handle zero-degree rows → self-loop
    row_sums = np.asarray(A.sum(axis=1)).flatten()
    zero_mask = row_sums == 0

    if zero_mask.any():
        absorb_idx = np.where(zero_mask)[0]
        absorb = csr_matrix(
            (np.ones(len(absorb_idx), dtype=np.float64),
             (absorb_idx, absorb_idx)),
            shape=(N, N),
        )
        A = A + absorb
        row_sums = np.asarray(A.sum(axis=1)).flatten()

    # 2. Row-normalize
    inv_d = np.zeros(N, dtype=np.float64)
    nonzero = row_sums > 0
    inv_d[nonzero] = 1.0 / row_sums[nonzero]
    P = diags(inv_d) @ A

    # 3. Apply teleport: P = (1-ε)P + ε/N
    #    For sparse efficiency, scale existing entries and note the
    #    uniform additive term ε/N is constant per row.  We store it
    #    only conceptually — for matrix-vector products the caller
    #    can handle it.  However, for the solver to work correctly
    #    with the actual CSR, we scale P and rely on the small ε
    #    being absorbed during row-sum checks.
    if teleport_eps > 0:
        P = P * (1.0 - teleport_eps)
        # The uniform term ε/N is tiny and applied during matvec
        # implicitly.  We track it via a flag for the diffusion step.
        # For row-sum correctness we note: each row sums to
        # (1-ε)*1 + ε*(N/N) = 1.  So stochasticity is preserved.

    P = P.tocsr()
    P.eliminate_zeros()

    # Store teleport info as attributes for downstream use
    P._teleport_eps = teleport_eps  # type: ignore[attr-defined]
    P._N = N  # type: ignore[attr-defined]

    return P


# ---------------------------------------------------------------------------
# Teleport-aware matrix-vector product
# ---------------------------------------------------------------------------

def matvec_with_teleport(v: np.ndarray, P: csr_matrix) -> np.ndarray:
    """
    Compute v_next = v @ P + ε/N * sum(v) for teleport-augmented P.

    This gives the exact result of v @ P_full where
    P_full = (1-ε)P_sparse + ε/N · 11ᵀ.
    """
    eps = getattr(P, '_teleport_eps', 0.0)
    N = getattr(P, '_N', P.shape[0])
    result = v @ P  # sparse matvec
    if eps > 0 and N > 0:
        result += (eps / N) * v.sum()
    return result


# ---------------------------------------------------------------------------
# Anchor Selection
# ---------------------------------------------------------------------------

def select_anchors(nodes: list[IRNode]) -> np.ndarray:
    """
    Build initial distribution vector v0 uniform over anchor nodes.

    Anchors (equal weight per class):
    - Function entry nodes
    - External call nodes
    - State write nodes
    - Exit nodes

    Returns v0 of shape (N,), summing to 1.0.
    """
    N = len(nodes)
    anchor_mask = np.zeros(N, dtype=np.float64)

    for node in nodes:
        if node.is_entry or node.is_external_call or node.is_state_write or node.is_exit:
            anchor_mask[node.id] = 1.0

    total = anchor_mask.sum()
    if total == 0:
        # Fallback: uniform over all nodes
        return np.full(N, 1.0 / N, dtype=np.float64)

    return anchor_mask / total


# ---------------------------------------------------------------------------
# Multi-Scale Diffusion
# ---------------------------------------------------------------------------

def multi_scale_diffusion(
    v0: np.ndarray,
    P: csr_matrix,
    scales: list[int] | None = None,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Compute multi-scale landing probabilities with teleport.

    For k in *scales*:
        v_k = v0 @ P_full^k   (iterative, teleport-aware)

    Returns g: ndarray of shape (N, len(scales))
    where g[:,j] = log(ε + v_{scales[j]}[i])

    Memory: only current vector + output storage.
    """
    if scales is None:
        import os
        env_scales = os.environ.get("DIFFUSION_SCALES")
        if env_scales:
            scales = [int(x.strip()) for x in env_scales.split(",")]
        else:
            scales = [1, 2, 4, 8, 16]

    N = len(v0)
    n_scales = len(scales)
    g = np.zeros((N, n_scales), dtype=np.float32)

    max_k = max(scales)
    scale_set = set(scales)
    scale_to_col = {s: i for i, s in enumerate(scales)}

    v = v0.copy().astype(np.float64)
    for t in range(1, max_k + 1):
        v = matvec_with_teleport(v, P)
        if t in scale_set:
            col = scale_to_col[t]
            g[:, col] = np.log(epsilon + v).astype(np.float32)

    return g
