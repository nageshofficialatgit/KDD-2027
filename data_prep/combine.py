#!/usr/bin/env python3
"""
Combine Module — Per-Channel Entropy/Resistance & λ_k Computation
-----------------------------------------------------------------
Computes stationary distributions, entropy, and Poisson resistance per
graph channel (CFG/DFG/Call), then combines them into data-driven mixing
weights λ_k using robust normalization (median/IQR) and harmonic mean.

Public API:
    compute_stationary(P, tol, maxiter)  → (pi, H, iters)
    compute_channel_stats(P_k, b)        → (H_k, R_k, status)
    compute_global_stats(H_all, R_all)   → dict
    compute_lambdas(H_k, R_k, global_stats, tau) → lambda_k
    compute_lambdas_degree(A_list)       → lambda_k  (fallback)
    mix_channels(A_list, lambda_k)       → A_total
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS_LOG = 1e-12        # avoid log(0)
_DELTA_LOG_R = 1e-9     # spec §9 delta for log(R_k + δ)
_LAMBDA_MIN = 0.05      # spec §9 clamp bounds
_LAMBDA_MAX = 0.85
_TAU_DEFAULT = 0.05     # spec §9 floor on R_hat
_CLAMP_STD = 3.0        # clamp standardised values to [-3, 3]
_MAP_LO = 0.01          # min-max target range
_MAP_HI = 0.99


# ---------------------------------------------------------------------------
# Stationary Distribution (Power Iteration)
# ---------------------------------------------------------------------------

def compute_stationary(
    P: csr_matrix,
    tol: float = 1e-9,
    maxiter: int = 200,
) -> tuple[np.ndarray, float, int]:
    """
    Compute stationary distribution π of row-stochastic P via power iteration.

    Parameters
    ----------
    P : csr_matrix
        Row-stochastic transition matrix (N×N).
    tol : float
        Convergence threshold on L1 norm of change.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    pi : ndarray, shape (N,)
        Stationary distribution (sums to 1).
    H : float
        Shannon entropy H = −Σ πᵢ log(πᵢ).
    iters : int
        Number of iterations until convergence.
    """
    N = P.shape[0]
    pi = np.full(N, 1.0 / N, dtype=np.float64)

    for it in range(1, maxiter + 1):
        pi_next = pi @ P  # 1×N sparse matvec
        # Ensure non-negative (numerical safety)
        np.maximum(pi_next, 0.0, out=pi_next)
        # Re-normalize to handle numerical drift
        s = pi_next.sum()
        if s > 0:
            pi_next /= s

        delta = np.abs(pi_next - pi).sum()
        pi = pi_next
        if delta < tol:
            break

    # Shannon entropy
    H = -np.sum(pi * np.log(pi + _EPS_LOG))

    log.debug("stationary converged in %d iters, delta=%.2e, H=%.4f", it, delta, H)
    return pi, H, it


# ---------------------------------------------------------------------------
# Per-Channel Stats
# ---------------------------------------------------------------------------

def compute_channel_stats(
    P_k: csr_matrix,
    b: np.ndarray,
    solve_fn=None,
) -> tuple[float, float, str]:
    """
    Compute entropy H_k and resistance R_k for one graph channel.

    Parameters
    ----------
    P_k : csr_matrix
        Row-stochastic transition matrix for channel k.
    b : ndarray
        Zero-mean pressure/source vector.
    solve_fn : callable, optional
        Poisson solver function with signature solve_fn(P, b) → (f, res, iters, status).
        If None, imports from poisson_solver.

    Returns
    -------
    H_k : float
        Shannon entropy of stationary distribution.
    R_k : float
        Poisson resistance = ‖f_k‖₂.
    status : str
        'ok' or description of failure.
    """
    # Stationary → entropy
    _, H_k, _ = compute_stationary(P_k)

    # Poisson → resistance
    if solve_fn is None:
        from poisson_solver import solve_poisson
        solve_fn = solve_poisson

    try:
        result = solve_fn(P_k, b)
        # Handle both old (just f) and new (f, res, iters, status) return
        if isinstance(result, tuple):
            f_k = result[0]
            status = result[3] if len(result) >= 4 else "ok"
        else:
            f_k = result
            status = "ok"

        R_k = float(np.linalg.norm(f_k, ord=2))
    except Exception as e:
        log.warning("Poisson solve failed for channel: %s", e)
        R_k = 0.0
        status = f"solve_failed:{str(e)[:40]}"

    return H_k, R_k, status


# ---------------------------------------------------------------------------
# Global Stats (Median/IQR across contracts)
# ---------------------------------------------------------------------------

def compute_global_stats(
    H_all: dict[str, np.ndarray],
    R_all: dict[str, np.ndarray],
) -> dict:
    """
    Compute robust normalisation statistics across a corpus.

    Parameters
    ----------
    H_all : dict
        {channel_name: array of H values, one per contract}.
    R_all : dict
        {channel_name: array of R values, one per contract}.

    Returns
    -------
    dict with keys:
        med_H, iqr_H, med_L, iqr_L — each a dict {channel_name: float}
    """
    stats: dict = {"med_H": {}, "iqr_H": {}, "med_L": {}, "iqr_L": {}}

    for ch in H_all:
        h_arr = np.asarray(H_all[ch], dtype=np.float64)
        r_arr = np.asarray(R_all[ch], dtype=np.float64)
        l_arr = np.log(r_arr + _DELTA_LOG_R)  # log-transform resistance

        stats["med_H"][ch] = float(np.median(h_arr))
        q75, q25 = np.percentile(h_arr, [75, 25])
        stats["iqr_H"][ch] = float(max(q75 - q25, 1e-8))  # floor to avoid /0

        stats["med_L"][ch] = float(np.median(l_arr))
        q75, q25 = np.percentile(l_arr, [75, 25])
        stats["iqr_L"][ch] = float(max(q75 - q25, 1e-8))

    return stats


# ---------------------------------------------------------------------------
# λ_k Computation
# ---------------------------------------------------------------------------

def compute_lambdas(
    H_k: dict[str, float],
    R_k: dict[str, float],
    global_stats: dict,
    tau: float = _TAU_DEFAULT,
) -> dict[str, float]:
    """
    Compute data-driven mixing weights λ_k from per-channel entropy/resistance.

    Steps (spec §9):
        1. Standardize H, L = log(R+δ) using corpus median/IQR
        2. Clamp to [-3, 3]
        3. Min-max across channels to [0.01, 0.99]
        4. Floor on R_hat: if R_hat < tau → s_k = 0
        5. s_k = harmonic_mean(H_hat, R_hat)
        6. λ_k = s_k / Σ s_j  (fallback to degree-norm if all zero)
        7. Clamp each λ_k to [0.05, 0.85], renormalize

    Parameters
    ----------
    H_k : dict {channel: float}
    R_k : dict {channel: float}
    global_stats : dict from compute_global_stats
    tau : float
        Floor threshold on R_hat.

    Returns
    -------
    dict {channel: float} with sum ≈ 1.0.
    """
    channels = sorted(H_k.keys())
    n = len(channels)
    if n == 0:
        return {}

    # Step 1: Standardize
    H_std = np.zeros(n, dtype=np.float64)
    L_std = np.zeros(n, dtype=np.float64)

    for i, ch in enumerate(channels):
        med_h = global_stats["med_H"].get(ch, 0.0)
        iqr_h = global_stats["iqr_H"].get(ch, 1.0)
        H_std[i] = (H_k[ch] - med_h) / iqr_h

        l_val = np.log(R_k[ch] + _DELTA_LOG_R)
        med_l = global_stats["med_L"].get(ch, 0.0)
        iqr_l = global_stats["iqr_L"].get(ch, 1.0)
        L_std[i] = (l_val - med_l) / iqr_l

    # Step 2: Clamp
    np.clip(H_std, -_CLAMP_STD, _CLAMP_STD, out=H_std)
    np.clip(L_std, -_CLAMP_STD, _CLAMP_STD, out=L_std)

    # Step 3: Min-max across channels to [0.01, 0.99]
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.full_like(arr, (_MAP_LO + _MAP_HI) / 2)
        scaled = (arr - lo) / (hi - lo)
        return _MAP_LO + scaled * (_MAP_HI - _MAP_LO)

    H_hat = _minmax(H_std)
    R_hat = _minmax(L_std)

    # Step 4 & 5: Floor + harmonic mean
    eps_hm = 1e-12
    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if R_hat[i] < tau:
            s[i] = 0.0
        else:
            s[i] = 2.0 * H_hat[i] * R_hat[i] / (H_hat[i] + R_hat[i] + eps_hm)

    # Step 6: Normalize
    s_sum = s.sum()
    if s_sum < 1e-12:
        # Fallback: equal weights
        log.warning("All s_k ≈ 0; falling back to equal λ_k")
        lam = np.full(n, 1.0 / n)
    else:
        lam = s / s_sum

    # Step 7: Clamp and renormalize (iterate to convergence)
    for _ in range(5):
        np.clip(lam, _LAMBDA_MIN, _LAMBDA_MAX, out=lam)
        lam /= lam.sum()

    result = {ch: float(lam[i]) for i, ch in enumerate(channels)}

    log.info("λ_k: %s  (s_k raw: %s)", result, {ch: float(s[i]) for i, ch in enumerate(channels)})
    return result


def compute_lambdas_degree(A_list: dict[str, csr_matrix]) -> dict[str, float]:
    """
    Degree-based fallback when no global stats are available (single-contract mode).

    λ_k ∝ total_degree(A_k) = sum of all edge weights.
    Clamped to [0.05, 0.85] and renormalized.

    Parameters
    ----------
    A_list : dict {channel_name: csr_matrix}

    Returns
    -------
    dict {channel_name: float} with sum ≈ 1.0.
    """
    channels = sorted(A_list.keys())
    n = len(channels)
    if n == 0:
        return {}

    degrees = np.zeros(n, dtype=np.float64)
    for i, ch in enumerate(channels):
        degrees[i] = float(A_list[ch].sum())

    total = degrees.sum()
    if total < 1e-12:
        lam = np.full(n, 1.0 / n)
    else:
        lam = degrees / total

    # Clamp and renormalize (iterate to convergence)
    for _ in range(5):
        np.clip(lam, _LAMBDA_MIN, _LAMBDA_MAX, out=lam)
        lam /= lam.sum()

    result = {ch: float(lam[i]) for i, ch in enumerate(channels)}
    log.info("λ_k (degree-based fallback): %s", result)
    return result


# ---------------------------------------------------------------------------
# Channel Mixing
# ---------------------------------------------------------------------------

def mix_channels(
    A_list: dict[str, csr_matrix],
    lambda_k: dict[str, float],
) -> csr_matrix:
    """
    Weighted sparse addition:  A_total = Σ_k λ_k · A_k.

    Parameters
    ----------
    A_list : dict {channel_name: csr_matrix}
    lambda_k : dict {channel_name: float}

    Returns
    -------
    A_total : csr_matrix
    """
    channels = sorted(A_list.keys())
    A_total = None

    for ch in channels:
        weighted = lambda_k[ch] * A_list[ch]
        if A_total is None:
            A_total = weighted.copy()
        else:
            A_total = A_total + weighted

    if A_total is None:
        raise ValueError("No channels provided to mix_channels")

    A_total.eliminate_zeros()
    return A_total.tocsr()
