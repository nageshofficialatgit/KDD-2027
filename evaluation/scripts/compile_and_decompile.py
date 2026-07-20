#!/usr/bin/env python3
"""
Tier B: Compile → Decompile Pipeline
Compiles SBCurated contracts to EVM bytecode, then decompiles via Panoramix.
Outputs pseudocode files + contract-level labels.
"""
import os, re, json, sys, signal, subprocess, traceback
from multiprocessing import Process, Queue

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
SOL_DIR = os.path.join(PROJECT_ROOT, "Dataset SmartContracts/SBCurated_141/SBCurated_141/SolidityFiles_Data")
GT_FILE = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/verified_sbcurated_ground_truth.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/decompiled")
OUT_SOL_DIR = os.path.join(OUT_DIR, "sol_files")
OUT_LABELS = os.path.join(OUT_DIR, "decompiled_labels.jsonl")
TIMEOUT = 45  # seconds per decompilation


def get_solc_version(source):
    """Extract pragma solidity version from source code."""
    m = re.search(r'pragma\s+solidity\s*[\^>=<]*\s*(0\.\d+\.\d+)', source)
    if m:
        return m.group(1)
    m = re.search(r'pragma\s+solidity\s*[\^>=<]*\s*(0\.\d+)', source)
    if m:
        # Default to latest patch
        major_minor = m.group(1)
        return major_minor + ".26" if "0.4" in major_minor else major_minor + ".0"
    return None


def compile_contract(sol_path, solc_version):
    """Compile a contract and return runtime bytecode."""
    # Switch solc version
    subprocess.run(["solc-select", "use", solc_version],
                    capture_output=True, text=True, timeout=10)

    result = subprocess.run(
        ["solc", "--bin-runtime", sol_path],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return None, result.stderr[:200]

    # Extract bytecode (last non-empty line that looks like hex)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and all(c in '0123456789abcdef' for c in line) and len(line) > 10:
            return line, None
    return None, "No bytecode found in output"


def _decompile_worker(bytecode, q):
    """Worker process for decompilation with timeout."""
    try:
        from panoramix.decompiler import decompile_bytecode
        result = decompile_bytecode(bytecode)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', result.text)
        q.put(("ok", clean))
    except Exception as e:
        q.put(("error", str(e)))


def decompile_bytecode_safe(bytecode, timeout=TIMEOUT):
    """Decompile with timeout using a subprocess to prevent hangs."""
    q = Queue()
    p = Process(target=_decompile_worker, args=(bytecode, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
        return None, "TIMEOUT"
    if not q.empty():
        status, data = q.get()
        if status == "ok":
            return data, None
        else:
            return None, data
    return None, "No result from worker"


def main():
    print("=== Tier B: Compile & Decompile SBCurated Contracts ===")
    os.makedirs(OUT_SOL_DIR, exist_ok=True)

    # Load ground truth
    gt = {}
    with open(GT_FILE, 'r') as f:
        for line in f:
            d = json.loads(line)
            gt[d["filename"]] = d

    sol_files = sorted([f for f in os.listdir(SOL_DIR) if f.endswith('.sol')])
    print(f"Found {len(sol_files)} .sol files")

    results = {
        "compiled": 0, "compile_fail": 0,
        "decompiled": 0, "decompile_fail": 0, "decompile_timeout": 0
    }
    labels_out = []

    for i, fn in enumerate(sol_files):
        filepath = os.path.join(SOL_DIR, fn)
        print(f"[{i+1}/{len(sol_files)}] {fn}...", end=" ", flush=True)

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()

            # Step 1: Get solc version
            version = get_solc_version(source)
            if not version:
                print("SKIP (no pragma)")
                results["compile_fail"] += 1
                continue

            # Step 2: Compile
            bytecode, err = compile_contract(filepath, version)
            if not bytecode:
                print(f"COMPILE_FAIL ({err[:50]})")
                results["compile_fail"] += 1
                continue
            results["compiled"] += 1

            # Step 3: Decompile
            pseudocode, err = decompile_bytecode_safe(bytecode)
            if not pseudocode:
                reason = "TIMEOUT" if err == "TIMEOUT" else f"DECOMPILE_FAIL ({err[:50]})"
                print(reason)
                if err == "TIMEOUT":
                    results["decompile_timeout"] += 1
                else:
                    results["decompile_fail"] += 1
                continue

            results["decompiled"] += 1

            # Save decompiled pseudocode
            out_path = os.path.join(OUT_SOL_DIR, fn + ".decompiled")
            with open(out_path, 'w') as f:
                f.write(pseudocode)

            # Create contract-level label entry
            vuln_types = gt[fn]["vulnerability_types"] if fn in gt else []
            is_vulnerable = 1 if (fn in gt and len(gt[fn]["vulnerable_lines"]) > 0) else 0
            label_entry = {
                "dataset": "SBCurated_Decompiled",
                "filename": fn,
                "source_code": pseudocode,
                "is_vulnerable": is_vulnerable,
                "vulnerability_types": vuln_types,
                "original_vulnerable_lines": gt[fn]["vulnerable_lines"] if fn in gt else [],
            }
            labels_out.append(label_entry)
            print(f"OK ({len(pseudocode)} chars)")

        except Exception as e:
            print(f"ERROR: {e}")
            results["compile_fail"] += 1

    # Save labels
    with open(OUT_LABELS, 'w') as f:
        for entry in labels_out:
            f.write(json.dumps(entry) + '\n')

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Compiled:           {results['compiled']}")
    print(f"  Compile failures:   {results['compile_fail']}")
    print(f"  Decompiled:         {results['decompiled']}")
    print(f"  Decompile failures: {results['decompile_fail']}")
    print(f"  Decompile timeouts: {results['decompile_timeout']}")
    print(f"  Labels saved to:    {OUT_LABELS}")
    print(f"  Pseudocode dir:     {OUT_SOL_DIR}/")


if __name__ == "__main__":
    main()
