#!/usr/bin/env python3
"""
Poisson Potential Solver
------------------------
Solves the system (I − P) f = b via BiCGSTAB with Jacobi preconditioner
and fallback chain (damping → LGMRES).  Returns diagnostics.

All operations use scipy.sparse.  No dense matrices.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import bicgstab, lgmres, LinearOperator

from graph_builder import IRNode

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pressure Vector
# ---------------------------------------------------------------------------

def build_pressure_vector(nodes: list[IRNode]) -> np.ndarray:
    """
    Build the source/sink pressure vector b.

    Sources (+1):  ENTRY nodes and EXTERNAL call-site nodes.
    Sinks   (−1):  STATE_WRITE nodes and EXIT nodes.

    Then normalize to zero mean: b -= mean(b).

    Returns b of shape (N,).
    """
    N = len(nodes)
    b = np.zeros(N, dtype=np.float64)

    for node in nodes:
        if node.is_entry or node.is_external_call:
            b[node.id] += 1.0
        if node.is_exit or node.is_state_write:
            b[node.id] -= 1.0

    # Ensure zero mean
    b -= b.mean()
    return b


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_poisson(
    P: csr_matrix,
    b: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    damping: float = 1e-8,
) -> tuple[np.ndarray, float, int, str]:
    """
    Solve  (I − P) f = b  using BiCGSTAB with Jacobi preconditioning.

    Fallback chain:
        1. BiCGSTAB on (I − P + αI)
        2. BiCGSTAB with damping: (I − (1-η)P + αI)
        3. LGMRES fallback

    Returns
    -------
    f : ndarray, shape (N,)
        Solution, centered to zero mean and unit std.
    residual : float
        Final ‖(I-P)f - b‖₂.
    iter_count : int
        Solver iterations (approximate).
    status : str
        'ok', 'ok_damped', 'ok_lgmres', or 'failed'.
    """
    N = P.shape[0]
    alpha = 1e-6  # regularization

    def _build_system(P_mat: csr_matrix) -> tuple[csr_matrix, LinearOperator]:
        """Build A_sys = I - P + αI and Jacobi preconditioner."""
        A_sys = eye(N, format='csr') - P_mat + alpha * eye(N, format='csr')
        diag_vals = np.asarray(A_sys.diagonal()).flatten()
        safe_diag = np.where(np.abs(diag_vals) > 1e-12, diag_vals, 1.0)
        M_inv_diag = 1.0 / safe_diag
        M_inv = LinearOperator(
            shape=(N, N),
            matvec=lambda x: M_inv_diag * x,
        )
        return A_sys, M_inv

    def _normalize(f: np.ndarray) -> np.ndarray:
        """Center to zero mean, normalize to unit variance."""
        f = f - f.mean()
        std = f.std()
        if std > 1e-12:
            f = f / std
        return f

    def _residual(A_sys, f, b):
        return float(np.linalg.norm(A_sys @ f - b, ord=2))

    # --- Attempt 1: BiCGSTAB ---
    A_sys, M_inv = _build_system(P)
    f, info = bicgstab(A_sys, b, M=M_inv, maxiter=max_iter, atol=tol)
    res = _residual(A_sys, f, b)

    if info == 0 and res < tol * 100:
        f = _normalize(f)
        log.debug("Poisson solved: bicgstab, residual=%.2e", res)
        return f, res, max_iter, "ok"

    # --- Attempt 2: BiCGSTAB with damping ---
    log.info("BiCGSTAB primary failed (info=%d, res=%.2e), trying damped...", info, res)
    P_damped = P * (1.0 - damping)
    A_sys_d, M_inv_d = _build_system(P_damped)
    f, info = bicgstab(A_sys_d, b, M=M_inv_d, maxiter=max_iter, atol=tol)
    res = _residual(A_sys_d, f, b)

    if info == 0 and res < tol * 100:
        f = _normalize(f)
        log.debug("Poisson solved: bicgstab damped, residual=%.2e", res)
        return f, res, max_iter, "ok_damped"

    # --- Attempt 3: LGMRES fallback ---
    log.info("BiCGSTAB damped failed (info=%d, res=%.2e), trying LGMRES...", info, res)
    try:
        f, info = lgmres(A_sys, b, maxiter=max_iter, atol=tol)
        res = _residual(A_sys, f, b)
        if info == 0:
            f = _normalize(f)
            log.debug("Poisson solved: lgmres, residual=%.2e", res)
            return f, res, max_iter, "ok_lgmres"
    except Exception as e:
        log.warning("LGMRES failed: %s", e)

    # --- All failed: return best-effort ---
    log.debug("All Poisson solvers failed; returning best-effort (res=%.2e)", res)
    f = _normalize(f)
    return f, res, max_iter, "failed"
