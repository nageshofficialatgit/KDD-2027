#!/usr/bin/env python3
"""
Graph Builder — Sparse CPG Construction via Slither IR
------------------------------------------------------
Parses a Solidity source file via Slither and constructs three sparse
adjacency matrices (CFG, DFG, Call Graph) over IR-level instruction nodes.

All matrices are scipy.sparse.csr_matrix.  No NetworkX. No dense storage.

Public API:
    build_cpg(sol_path, solc_path) -> CPGResult | None
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    """One IR-level instruction node."""
    id: int
    function_name: str
    opcode: str                              # e.g. "CONDITION", "EXPRESSION", "RETURN"
    source_span: tuple[int, int]             # (start_token_idx, end_token_idx), (-1,-1) if synthetic
    is_entry: bool
    is_exit: bool
    is_external_call: bool = False
    is_state_write: bool = False
    contract_name: str = ""
    expression_str: str = ""
    variables_read: list[str] = field(default_factory=list)
    variables_written: list[str] = field(default_factory=list)
    source_lines: list[int] = field(default_factory=list)


@dataclass
class CPGResult:
    """Container for the complete CPG of a single contract."""
    nodes: list[IRNode]
    A_cfg: csr_matrix
    A_dfg: csr_matrix
    A_call: csr_matrix
    A_unified: csr_matrix
    source_code: str


# ---------------------------------------------------------------------------
# Slither Analysis → IR Nodes
# ---------------------------------------------------------------------------

def _has_external_call(node) -> bool:
    """Check if a Slither CFG node contains an external call."""
    try:
        from slither.core.expressions import CallExpression
        if hasattr(node, 'expression') and isinstance(node.expression, CallExpression):
            # Check if it's a low-level call (.call, .send, .transfer, .delegatecall)
            expr_str = str(node.expression)
            if any(kw in expr_str for kw in ('.call', '.send', '.transfer', '.delegatecall', '.staticcall')):
                return True
        # Also check high_level_calls and low_level_calls
        if hasattr(node, 'low_level_calls') and node.low_level_calls:
            return True
        if hasattr(node, 'external_calls_as_expressions') and node.external_calls_as_expressions:
            return True
    except Exception:
        pass
    return False


def _has_state_write(node) -> bool:
    """Check if a Slither CFG node writes to contract state."""
    try:
        if hasattr(node, 'state_variables_written') and node.state_variables_written:
            return True
        if hasattr(node, 'variables_written'):
            from slither.core.variables.state_variable import StateVariable
            for var in node.variables_written:
                if isinstance(var, StateVariable):
                    return True
    except Exception:
        pass
    return False


def _extract_var_names(var_list) -> list[str]:
    """Extract variable name strings from a list of Slither variable objects."""
    names: list[str] = []
    try:
        for var in var_list:
            if hasattr(var, 'name') and var.name:
                names.append(var.name)
            elif hasattr(var, 'canonical_name') and var.canonical_name:
                names.append(var.canonical_name)
    except Exception:
        pass
    return names


def parse_contract(
    sol_path: str,
    solc_path: Optional[str] = None,
) -> Optional[tuple[list[IRNode], object]]:
    """
    Parse a Solidity file with Slither and extract IR-level nodes.

    Returns (nodes, slither_obj) or None on failure.
    """
    try:
        from slither import Slither
    except ImportError:
        raise ImportError("slither-analyzer is required: pip install slither-analyzer")

    try:
        slither_obj = Slither(sol_path, solc=solc_path)
    except Exception:
        return None

    nodes: list[IRNode] = []
    node_id = 0
    # Map from (slither_node_id, function_canonical_name) → our sequential id
    slither_to_id: dict[tuple[int, str], int] = {}

    for contract in slither_obj.contracts:
        for function in contract.functions:
            if not function.nodes:
                continue

            func_name = f"{contract.name}.{function.name}"
            func_nodes = list(function.nodes)

            for i, snode in enumerate(func_nodes):
                # Determine opcode from Slither node type
                opcode = str(snode.type) if hasattr(snode, 'type') else "UNKNOWN"

                # Expression string
                expr_str = ""
                if hasattr(snode, 'expression') and snode.expression:
                    expr_str = str(snode.expression).replace('\n', ' ').strip()

                # Source lines
                src_lines: list[int] = []
                if hasattr(snode, 'source_mapping') and snode.source_mapping:
                    sm = snode.source_mapping
                    if hasattr(sm, 'lines') and sm.lines:
                        src_lines = list(sm.lines)

                # Is entry / exit
                is_entry = (i == 0)  # first node of function
                is_exit = (len(list(snode.sons)) == 0) if hasattr(snode, 'sons') else (i == len(func_nodes) - 1)

                ir_node = IRNode(
                    id=node_id,
                    function_name=func_name,
                    opcode=opcode,
                    source_span=(-1, -1),  # filled later during token mapping
                    is_entry=is_entry,
                    is_exit=is_exit,
                    is_external_call=_has_external_call(snode),
                    is_state_write=_has_state_write(snode),
                    contract_name=contract.name,
                    expression_str=expr_str,
                    variables_read=_extract_var_names(
                        getattr(snode, 'variables_read', [])
                    ),
                    variables_written=_extract_var_names(
                        getattr(snode, 'variables_written', [])
                    ),
                    source_lines=src_lines,
                )
                nodes.append(ir_node)
                slither_to_id[(snode.node_id, func_name)] = node_id
                node_id += 1

    if not nodes:
        return None

    return nodes, slither_obj, slither_to_id


# ---------------------------------------------------------------------------
# Adjacency Matrix Builders
# ---------------------------------------------------------------------------

def build_cfg(
    nodes: list[IRNode],
    slither_obj,
    slither_to_id: dict[tuple[int, str], int],
) -> csr_matrix:
    """
    Build CFG adjacency matrix.
    Edge i → j if execution can transfer from instruction i to j.
    Weight = 1 / out_degree(i).
    """
    N = len(nodes)
    lil = lil_matrix((N, N), dtype=np.float64)

    for contract in slither_obj.contracts:
        for function in contract.functions:
            if not function.nodes:
                continue
            func_name = f"{contract.name}.{function.name}"

            for snode in function.nodes:
                key_src = (snode.node_id, func_name)
                if key_src not in slither_to_id:
                    continue
                src = slither_to_id[key_src]

                successors: list[int] = []

                # sons = direct successors in CFG
                if hasattr(snode, 'sons'):
                    for son in snode.sons:
                        key_dst = (son.node_id, func_name)
                        if key_dst in slither_to_id:
                            successors.append(slither_to_id[key_dst])

                if not successors:
                    continue

                w = 1.0 / len(successors)
                for dst in successors:
                    lil[src, dst] = w

    return lil.tocsr()


def build_dfg(
    nodes: list[IRNode],
    slither_obj,
    slither_to_id: dict[tuple[int, str], int],
) -> csr_matrix:
    """
    Build Data Flow Graph adjacency matrix.
    Edge def(v)@i → use(v)@j.
    Weight = 1 / number_of_uses(v) within the function.
    """
    N = len(nodes)
    lil = lil_matrix((N, N), dtype=np.float64)

    for contract in slither_obj.contracts:
        for function in contract.functions:
            if not function.nodes:
                continue
            func_name = f"{contract.name}.{function.name}"

            # Phase 1: collect writes per variable
            # {var_name: [list of our node ids that write it]}
            writes: dict[str, list[int]] = {}
            for snode in function.nodes:
                key = (snode.node_id, func_name)
                if key not in slither_to_id:
                    continue
                nid = slither_to_id[key]
                if hasattr(snode, 'variables_written'):
                    for var in snode.variables_written:
                        vname = getattr(var, 'name', None) or getattr(var, 'canonical_name', None)
                        if vname:
                            writes.setdefault(vname, []).append(nid)

            # Phase 2: for each read, count total uses of that variable
            # Then add weighted edges
            use_counts: dict[str, int] = {}
            for snode in function.nodes:
                key = (snode.node_id, func_name)
                if key not in slither_to_id:
                    continue
                if hasattr(snode, 'variables_read'):
                    for var in snode.variables_read:
                        vname = getattr(var, 'name', None) or getattr(var, 'canonical_name', None)
                        if vname and vname in writes:
                            use_counts[vname] = use_counts.get(vname, 0) + 1

            # Phase 3: add edges
            for snode in function.nodes:
                key = (snode.node_id, func_name)
                if key not in slither_to_id:
                    continue
                nid = slither_to_id[key]
                if hasattr(snode, 'variables_read'):
                    for var in snode.variables_read:
                        vname = getattr(var, 'name', None) or getattr(var, 'canonical_name', None)
                        if vname and vname in writes:
                            n_uses = max(use_counts.get(vname, 1), 1)
                            w = 1.0 / n_uses
                            for write_nid in writes[vname]:
                                if write_nid != nid:  # no self-loops
                                    lil[write_nid, nid] = w

    return lil.tocsr()


def build_call_graph(
    nodes: list[IRNode],
    slither_obj,
    slither_to_id: dict[tuple[int, str], int],
) -> csr_matrix:
    """
    Build intra-contract Call Graph adjacency matrix.
    call_site → callee_entry  (weight 1.0)
    callee_exit → successor(call_site)  (weight 1.0)
    """
    N = len(nodes)
    lil = lil_matrix((N, N), dtype=np.float64)

    # Build a map: function canonical name → (entry_node_id, exit_node_ids)
    func_entry_exit: dict[str, tuple[int, list[int]]] = {}
    for contract in slither_obj.contracts:
        for function in contract.functions:
            if not function.nodes:
                continue
            func_name = f"{contract.name}.{function.name}"
            func_snodes = list(function.nodes)
            if not func_snodes:
                continue

            entry_key = (func_snodes[0].node_id, func_name)
            entry_id = slither_to_id.get(entry_key)
            if entry_id is None:
                continue

            exit_ids: list[int] = []
            for snode in func_snodes:
                key = (snode.node_id, func_name)
                nid = slither_to_id.get(key)
                if nid is not None and nodes[nid].is_exit:
                    exit_ids.append(nid)

            func_entry_exit[func_name] = (entry_id, exit_ids)

    # Find call sites and add edges
    try:
        from slither.core.expressions import CallExpression
    except ImportError:
        return lil.tocsr()

    for contract in slither_obj.contracts:
        for function in contract.functions:
            if not function.nodes:
                continue
            func_name = f"{contract.name}.{function.name}"

            for snode in function.nodes:
                key_src = (snode.node_id, func_name)
                if key_src not in slither_to_id:
                    continue
                src = slither_to_id[key_src]

                # Check for internal function calls
                if hasattr(snode, 'internal_calls'):
                    for called_func in snode.internal_calls:
                        if not hasattr(called_func, 'name'):
                            continue
                        # Try to find the callee in our contract
                        callee_name = f"{contract.name}.{called_func.name}"
                        if callee_name in func_entry_exit:
                            callee_entry, callee_exits = func_entry_exit[callee_name]
                            # call_site → callee_entry
                            lil[src, callee_entry] = 1.0

                            # callee_exit → successor(call_site)
                            # Find successors of src in CFG
                            if hasattr(snode, 'sons'):
                                for son in snode.sons:
                                    son_key = (son.node_id, func_name)
                                    if son_key in slither_to_id:
                                        for exit_id in callee_exits:
                                            lil[exit_id, slither_to_id[son_key]] = 1.0

    return lil.tocsr()


def build_unified(
    A_cfg: csr_matrix,
    A_dfg: csr_matrix,
    A_call: csr_matrix,
    lambda_cfg: float = 1.0,
    lambda_dfg: float = 0.7,
    lambda_call: float = 0.5,
) -> csr_matrix:
    """
    Unified weighted adjacency:  A = λ_cfg·A_cfg + λ_dfg·A_dfg + λ_call·A_call
    """
    A = lambda_cfg * A_cfg + lambda_dfg * A_dfg + lambda_call * A_call
    A.eliminate_zeros()
    return A.tocsr()


# ---------------------------------------------------------------------------
# Top-Level Builder
# ---------------------------------------------------------------------------

def build_cpg(
    sol_path: str,
    solc_path: Optional[str] = None,
) -> Optional[CPGResult]:
    """
    Build a complete sparse CPG from a Solidity file.

    Parameters
    ----------
    sol_path : str
        Path to the .sol file.
    solc_path : str, optional
        Absolute path to the solc binary. If None, Slither uses default.

    Returns
    -------
    CPGResult or None on failure.
    """
    result = parse_contract(sol_path, solc_path)
    if result is None:
        return None
    nodes, slither_obj, slither_to_id = result

    if len(nodes) == 0:
        return None

    # Read source code
    try:
        with open(sol_path, 'r', encoding='utf-8', errors='ignore') as f:
            source_code = f.read()
    except Exception:
        source_code = ""

    A_cfg = build_cfg(nodes, slither_obj, slither_to_id)
    A_dfg = build_dfg(nodes, slither_obj, slither_to_id)
    A_call = build_call_graph(nodes, slither_obj, slither_to_id)
    A_unified = build_unified(A_cfg, A_dfg, A_call)

    return CPGResult(
        nodes=nodes,
        A_cfg=A_cfg,
        A_dfg=A_dfg,
        A_call=A_call,
        A_unified=A_unified,
        source_code=source_code,
    )


def build_cpg_from_source(
    source_code: str,
    solc_path: Optional[str] = None,
    tmp_dir: Optional[str] = None,
) -> Optional[CPGResult]:
    """
    Build CPG from a source code string (writes to temp file, analyses, cleans up).
    """
    import uuid
    td = tmp_dir or tempfile.gettempdir()
    tmp_file = os.path.join(td, f"_cpg_{uuid.uuid4().hex[:12]}.sol")
    try:
        with open(tmp_file, 'w', encoding='utf-8') as f:
            f.write(source_code)
        result = build_cpg(tmp_file, solc_path)
        return result
    finally:
        try:
            os.unlink(tmp_file)
        except OSError:
            pass
