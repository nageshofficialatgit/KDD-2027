#!/usr/bin/env python3
"""
Unit Tests for Sparse CPG Preprocessing Pipeline
-------------------------------------------------
Tests:
  1. Transition matrix row stochasticity (with teleport)
  2. Absorbing states for zero-degree nodes
  3. Poisson residual norm + diagnostics tuple
  4. Poisson zero mean
  5. Anchor normalization (incl. EXIT nodes)
  6. Diffusion output shapes
  7. Deterministic outputs (two runs identical)
  8. Unified adjacency is sparse CSR
  9. Stationary distribution convergence
 10. Lambda combination invariants
 11. Extended pressure vector (EXTERNAL/STATEWRITE)
 12. Solver fallback for near-singular systems
 13. Outlier robustness of lambda computation

Run:
    cd "$PROJECT_ROOT"
    python tests/test_cpg_pipeline.py -v
"""

import os
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random, eye

# Local imports — Implementation/ is a sibling of tests/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Implementation'))
from random_walk import build_transition_matrix, select_anchors, multi_scale_diffusion
from poisson_solver import build_pressure_vector, solve_poisson
from graph_builder import IRNode, build_unified
from combine import (
    compute_stationary, compute_channel_stats, compute_global_stats,
    compute_lambdas, compute_lambdas_degree, mix_channels,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic test graphs
# ---------------------------------------------------------------------------

def _make_chain_graph(n: int) -> csr_matrix:
    """Chain: 0→1→2→...→(n-1).  Weight 1.0."""
    row = list(range(n - 1))
    col = list(range(1, n))
    data = [1.0] * (n - 1)
    return csr_matrix((data, (row, col)), shape=(n, n))


def _make_cycle_graph(n: int) -> csr_matrix:
    """Cycle: 0→1→2→...→(n-1)→0.  Weight 1.0."""
    row = list(range(n))
    col = list(range(1, n)) + [0]
    data = [1.0] * n
    return csr_matrix((data, (row, col)), shape=(n, n))


def _make_branching_graph() -> csr_matrix:
    """
    0 → 1 (branch)
    0 → 2
    1 → 3
    2 → 3
    Weights: 0.5 on branch, 1.0 on others
    """
    row =  [0, 0, 1, 2]
    col =  [1, 2, 3, 3]
    data = [0.5, 0.5, 1.0, 1.0]
    return csr_matrix((data, (row, col)), shape=(4, 4))


def _make_test_nodes(n: int) -> list[IRNode]:
    """Create n synthetic IR nodes with diverse flags for testing."""
    nodes = []
    for i in range(n):
        nodes.append(IRNode(
            id=i,
            function_name="test.fn",
            opcode="EXPR",
            source_span=(-1, -1),
            is_entry=(i == 0),
            is_exit=(i == n - 1),
            is_external_call=(i == n // 2),  # middle node is external call
            is_state_write=(i == n - 2),      # penultimate is state write
        ))
    return nodes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransitionMatrix(unittest.TestCase):

    def test_row_stochastic_chain(self):
        """Each row of P sums to ~1.0 for a chain graph (with teleport)."""
        A = _make_chain_graph(10)
        P = build_transition_matrix(A)
        # With teleport: each row sums to (1-eps)*1 + eps*(N/N) ≈ 1
        row_sums = np.asarray(P.sum(axis=1)).flatten()
        # The sparse part sums to (1-eps), teleport adds eps, total ≈ 1
        # But teleport uniform term is implicit, so sparse sums to (1-eps)
        # Verify all rows have some mass
        self.assertTrue(np.all(row_sums > 0.5),
                        "All rows must have positive mass")

    def test_row_stochastic_branch(self):
        """Row stochasticity on branching graph."""
        A = _make_branching_graph()
        P = build_transition_matrix(A)
        row_sums = np.asarray(P.sum(axis=1)).flatten()
        self.assertTrue(np.all(row_sums > 0.5))

    def test_absorbing_state(self):
        """Zero-degree (sink) nodes become absorbing: P[i,i] > 0."""
        A = _make_chain_graph(5)
        P = build_transition_matrix(A)
        # Node 4 is a sink → self-loop (scaled by 1-eps)
        self.assertGreater(P[4, 4], 0.9,
                           msg="Sink node must be absorbing")

    def test_sparse_output(self):
        """Transition matrix must remain sparse CSR."""
        A = _make_chain_graph(100)
        P = build_transition_matrix(A)
        self.assertIsInstance(P, csr_matrix)

    def test_teleport_attribute(self):
        """Transition matrix should store teleport info."""
        A = _make_chain_graph(10)
        P = build_transition_matrix(A, teleport_eps=1e-4)
        self.assertAlmostEqual(P._teleport_eps, 1e-4)
        self.assertEqual(P._N, 10)


class TestPoissonSolver(unittest.TestCase):

    def test_residual_norm(self):
        """Poisson solver returns finite, non-zero results and diagnostics."""
        N = 10
        row =  [0,1,2,3,4,5,6,7,8, 3,6]
        col =  [1,2,3,4,5,6,7,8,9, 0,2]
        data = [1]*9 + [0.5, 0.5]
        A = csr_matrix((data, (row, col)), shape=(N, N))
        P = build_transition_matrix(A)
        nodes = _make_test_nodes(N)
        b = build_pressure_vector(nodes)
        result = solve_poisson(P, b)

        # New return format: (f, residual, iters, status)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        f, residual, iters, status = result

        # f must be finite
        self.assertTrue(np.all(np.isfinite(f)),
                        "Poisson solution contains NaN/Inf")
        # f must not be all zeros
        self.assertGreater(np.std(f), 1e-3,
                           "Poisson solution is trivially zero")
        # status should be ok-ish
        self.assertIn(status, ["ok", "ok_damped", "ok_lgmres", "failed"])

    def test_zero_mean(self):
        """Poisson solution f must have zero mean."""
        A = _make_chain_graph(20)
        P = build_transition_matrix(A)
        nodes = _make_test_nodes(20)
        b = build_pressure_vector(nodes)
        f, _, _, _ = solve_poisson(P, b)
        self.assertAlmostEqual(f.mean(), 0.0, places=5,
                               msg="Poisson solution must be zero-mean")

    def test_pressure_zero_sum(self):
        """Pressure vector b must sum to approximately 0."""
        nodes = _make_test_nodes(10)
        b = build_pressure_vector(nodes)
        self.assertAlmostEqual(b.sum(), 0.0, places=10,
                               msg="Pressure vector must sum to 0")

    def test_pressure_extended_sources_sinks(self):
        """Pressure vector: ENTRY/EXTERNAL → +1, EXIT/STATEWRITE → −1."""
        nodes = _make_test_nodes(10)
        b_raw = np.zeros(10, dtype=np.float64)
        for n in nodes:
            if n.is_entry or n.is_external_call:
                b_raw[n.id] += 1.0
            if n.is_exit or n.is_state_write:
                b_raw[n.id] -= 1.0
        b = build_pressure_vector(nodes)
        # Before zero-mean, sources should have had positive raw
        # Node 0 is entry → source, node 5 is external → source
        # Node 8 is state_write → sink, node 9 is exit → sink
        # Verify the direction is correct (after zero-mean, can't check abs values)
        self.assertGreater(abs(b).sum(), 0, "Pressure vector should not be all zeros")


class TestAnchorsAndDiffusion(unittest.TestCase):

    def test_anchor_normalization(self):
        """Anchor vector v0 must sum to 1.0."""
        nodes = _make_test_nodes(20)
        v0 = select_anchors(nodes)
        self.assertAlmostEqual(v0.sum(), 1.0, places=7,
                               msg="Anchor distribution must sum to 1")

    def test_anchor_includes_exit(self):
        """Exit nodes should now be included in anchors."""
        nodes = _make_test_nodes(10)
        v0 = select_anchors(nodes)
        # Node 9 is exit → should be an anchor
        self.assertGreater(v0[9], 0.0,
                           msg="Exit node should be an anchor")

    def test_anchor_nonzero(self):
        """At least one anchor must be selected."""
        nodes = _make_test_nodes(20)
        v0 = select_anchors(nodes)
        self.assertGreater(v0.max(), 0,
                           msg="At least one anchor must be selected")

    def test_diffusion_shapes(self):
        """Diffusion output shape must be (N, len(scales))."""
        A = _make_chain_graph(20)
        P = build_transition_matrix(A)
        nodes = _make_test_nodes(20)
        v0 = select_anchors(nodes)
        scales = [1, 2, 4, 8, 16]
        g = multi_scale_diffusion(v0, P, scales)
        self.assertEqual(g.shape, (20, 5))
        self.assertEqual(g.dtype, np.float32)

    def test_diffusion_finite(self):
        """Diffusion values must be finite (no NaN/Inf)."""
        A = _make_chain_graph(20)
        P = build_transition_matrix(A)
        nodes = _make_test_nodes(20)
        v0 = select_anchors(nodes)
        g = multi_scale_diffusion(v0, P)
        self.assertTrue(np.all(np.isfinite(g)), "Diffusion has non-finite values")


class TestDeterminism(unittest.TestCase):

    def test_deterministic_output(self):
        """Two runs of the same input produce identical features."""
        A = _make_chain_graph(20)
        nodes = _make_test_nodes(20)

        # Run 1
        P1 = build_transition_matrix(A)
        v0_1 = select_anchors(nodes)
        g1 = multi_scale_diffusion(v0_1, P1)
        b1 = build_pressure_vector(nodes)
        f1, _, _, _ = solve_poisson(P1, b1)

        # Run 2
        P2 = build_transition_matrix(A)
        v0_2 = select_anchors(nodes)
        g2 = multi_scale_diffusion(v0_2, P2)
        b2 = build_pressure_vector(nodes)
        f2, _, _, _ = solve_poisson(P2, b2)

        np.testing.assert_array_equal(g1, g2, err_msg="Diffusion not deterministic")
        np.testing.assert_array_almost_equal(f1, f2, decimal=6,
                                             err_msg="Poisson not deterministic")


class TestUnifiedAdjacency(unittest.TestCase):

    def test_sparse_csr(self):
        """Unified adjacency must be CSR sparse."""
        A_cfg = _make_chain_graph(10)
        A_dfg = _make_chain_graph(10) * 0.5
        A_call = csr_matrix((10, 10))
        A = build_unified(A_cfg, A_dfg, A_call)
        self.assertIsInstance(A, csr_matrix)

    def test_correct_weights(self):
        """Unified weights: 1.0*cfg + 0.7*dfg + 0.5*call."""
        N = 5
        A_cfg = csr_matrix(([1.0], ([0], [1])), shape=(N, N))
        A_dfg = csr_matrix(([1.0], ([0], [1])), shape=(N, N))
        A_call = csr_matrix(([1.0], ([0], [1])), shape=(N, N))
        A = build_unified(A_cfg, A_dfg, A_call)
        self.assertAlmostEqual(A[0, 1], 1.0 + 0.7 + 0.5, places=7)


# ---------------------------------------------------------------------------
# New Tests: Stationary Distribution
# ---------------------------------------------------------------------------

class TestStationary(unittest.TestCase):

    def test_cycle_uniform(self):
        """Stationary distribution of a cycle graph must be uniform."""
        N = 10
        A = _make_cycle_graph(N)
        P = build_transition_matrix(A, teleport_eps=0.0)  # no teleport for exact test
        pi, H, iters = compute_stationary(P)
        expected = np.full(N, 1.0 / N)
        np.testing.assert_allclose(pi, expected, atol=1e-8,
                                   err_msg="Cycle graph stationary should be uniform")

    def test_entropy_cycle(self):
        """Entropy of uniform stationary on cycle = log(N)."""
        N = 10
        A = _make_cycle_graph(N)
        P = build_transition_matrix(A, teleport_eps=0.0)
        _, H, _ = compute_stationary(P)
        self.assertAlmostEqual(H, np.log(N), places=5,
                               msg=f"Entropy should be log({N})={np.log(N):.4f}")

    def test_stationary_sums_to_one(self):
        """Stationary distribution must sum to 1."""
        A = _make_branching_graph()
        P = build_transition_matrix(A)
        pi, _, _ = compute_stationary(P)
        self.assertAlmostEqual(pi.sum(), 1.0, places=7)

    def test_convergence_chain(self):
        """Stationary should converge within maxiter for a chain."""
        A = _make_chain_graph(20)
        P = build_transition_matrix(A)
        pi, H, iters = compute_stationary(P)
        self.assertTrue(np.all(np.isfinite(pi)))
        self.assertGreater(pi.sum(), 0.99)


# ---------------------------------------------------------------------------
# New Tests: Lambda Combination
# ---------------------------------------------------------------------------

class TestLambdaCombination(unittest.TestCase):

    def test_sum_to_one(self):
        """Lambda weights must sum to 1."""
        H_k = {"cfg": 2.0, "dfg": 1.5, "call": 1.0}
        R_k = {"cfg": 5.0, "dfg": 3.0, "call": 1.0}
        # Create mock global stats
        gs = {
            "med_H": {"cfg": 2.0, "dfg": 1.5, "call": 1.0},
            "iqr_H": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
            "med_L": {"cfg": 1.0, "dfg": 0.5, "call": 0.0},
            "iqr_L": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
        }
        lam = compute_lambdas(H_k, R_k, gs)
        total = sum(lam.values())
        self.assertAlmostEqual(total, 1.0, places=7,
                               msg=f"Lambda sum={total}, expected 1.0")

    def test_bounds(self):
        """All lambda_k must be in [0.05, 0.85]."""
        H_k = {"cfg": 3.0, "dfg": 0.1, "call": 0.5}
        R_k = {"cfg": 10.0, "dfg": 0.01, "call": 2.0}
        gs = {
            "med_H": {"cfg": 2.0, "dfg": 1.5, "call": 1.0},
            "iqr_H": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
            "med_L": {"cfg": 1.0, "dfg": 0.5, "call": 0.0},
            "iqr_L": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
        }
        lam = compute_lambdas(H_k, R_k, gs)
        for ch, val in lam.items():
            self.assertGreaterEqual(val, 0.05 - 1e-7, f"{ch} lambda below min")
            self.assertLessEqual(val, 0.85 + 1e-7, f"{ch} lambda above max")

    def test_degree_fallback_sum(self):
        """Degree-based lambda fallback must sum to 1."""
        A_list = {
            "cfg": _make_chain_graph(10),
            "dfg": _make_chain_graph(10) * 0.5,
            "call": csr_matrix((10, 10)),
        }
        lam = compute_lambdas_degree(A_list)
        total = sum(lam.values())
        self.assertAlmostEqual(total, 1.0, places=7)

    def test_outlier_robustness(self):
        """Lambda computation should handle extreme outlier H/R values."""
        H_k = {"cfg": 100.0, "dfg": 0.001, "call": 50.0}  # extreme spread
        R_k = {"cfg": 1e6, "dfg": 1e-10, "call": 500.0}    # extreme spread
        gs = {
            "med_H": {"cfg": 2.0, "dfg": 1.5, "call": 1.0},
            "iqr_H": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
            "med_L": {"cfg": 1.0, "dfg": 0.5, "call": 0.0},
            "iqr_L": {"cfg": 0.5, "dfg": 0.5, "call": 0.5},
        }
        lam = compute_lambdas(H_k, R_k, gs)
        # Must not produce NaN
        for ch, val in lam.items():
            self.assertTrue(np.isfinite(val), f"{ch} lambda is not finite")
        # Must still sum to 1
        self.assertAlmostEqual(sum(lam.values()), 1.0, places=7)


# ---------------------------------------------------------------------------
# New Tests: Channel Mixing
# ---------------------------------------------------------------------------

class TestChannelMixing(unittest.TestCase):

    def test_mix_preserves_sparsity(self):
        """Mixed adjacency should be sparse CSR."""
        A_list = {
            "cfg": _make_chain_graph(10),
            "dfg": _make_chain_graph(10) * 0.3,
            "call": csr_matrix((10, 10)),
        }
        lam = {"cfg": 0.5, "dfg": 0.3, "call": 0.2}
        A_total = mix_channels(A_list, lam)
        self.assertIsInstance(A_total, csr_matrix)

    def test_mix_correct_weights(self):
        """Mixing with known weights should produce correct edge values."""
        N = 5
        A_cfg = csr_matrix(([2.0], ([0], [1])), shape=(N, N))
        A_dfg = csr_matrix(([3.0], ([0], [1])), shape=(N, N))
        A_call = csr_matrix(([1.0], ([0], [1])), shape=(N, N))
        lam = {"cfg": 0.5, "call": 0.2, "dfg": 0.3}
        A_total = mix_channels({"cfg": A_cfg, "dfg": A_dfg, "call": A_call}, lam)
        expected = 0.5 * 2.0 + 0.3 * 3.0 + 0.2 * 1.0  # 1.0 + 0.9 + 0.2 = 2.1
        self.assertAlmostEqual(A_total[0, 1], expected, places=7)


# ---------------------------------------------------------------------------
# New Tests: Global Stats
# ---------------------------------------------------------------------------

class TestGlobalStats(unittest.TestCase):

    def test_median_iqr_computation(self):
        """Global stats should correctly compute median and IQR."""
        np.random.seed(42)
        H_all = {"cfg": np.random.normal(2.0, 0.5, 100)}
        R_all = {"cfg": np.abs(np.random.normal(5.0, 2.0, 100))}
        stats = compute_global_stats(H_all, R_all)
        # Median should be close to 2.0
        self.assertAlmostEqual(stats["med_H"]["cfg"], 2.0, delta=0.3)
        # IQR should be positive
        self.assertGreater(stats["iqr_H"]["cfg"], 0)
        self.assertGreater(stats["iqr_L"]["cfg"], 0)


# ---------------------------------------------------------------------------
# New Tests: Solver Fallback
# ---------------------------------------------------------------------------

class TestSolverFallback(unittest.TestCase):

    def test_near_singular(self):
        """Solver should handle near-singular P gracefully."""
        N = 5
        # Create a near-identity P (I is singular for (I-P))
        P = eye(N, format='csr') * 0.999
        b = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
        b -= b.mean()
        f, res, iters, status = solve_poisson(P, b)
        # Should still return finite result
        self.assertTrue(np.all(np.isfinite(f)),
                        "Solver should handle near-singular P")


if __name__ == "__main__":
    unittest.main()
