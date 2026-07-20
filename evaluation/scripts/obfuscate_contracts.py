#!/usr/bin/env python3
"""
Tier A: Source-Level Obfuscation Pipeline
Strips comments, renames identifiers, normalizes whitespace.
Preserves line count so ground truth line numbers transfer directly.
"""
import os, re, json, sys

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
SOL_DIR = os.path.join(PROJECT_ROOT, "Dataset SmartContracts/SBCurated_141/SBCurated_141/SolidityFiles_Data")
GT_FILE = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/verified_sbcurated_ground_truth.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/data/obfuscated")
OUT_SOL_DIR = os.path.join(OUT_DIR, "sol_files")
OUT_LABELS = os.path.join(OUT_DIR, "obfuscated_labels.jsonl")
OUT_ANNOS = os.path.join(OUT_DIR, "obfuscated_annotations.jsonl")

# Solidity keywords that must NOT be renamed
SOLIDITY_KEYWORDS = {
    "pragma", "solidity", "contract", "library", "interface", "is", "using",
    "function", "modifier", "event", "struct", "enum", "mapping",
    "public", "private", "internal", "external", "pure", "view", "payable",
    "constant", "immutable", "override", "virtual", "abstract",
    "returns", "return", "if", "else", "for", "while", "do", "break", "continue",
    "require", "assert", "revert", "emit", "new", "delete", "throw",
    "true", "false", "this", "super", "selfdestruct", "suicide",
    "msg", "block", "tx", "abi", "now",
    "address", "bool", "string", "bytes", "byte",
    "uint", "int", "uint8", "uint16", "uint32", "uint64", "uint128", "uint256",
    "int8", "int16", "int32", "int64", "int128", "int256",
    "bytes1", "bytes2", "bytes4", "bytes8", "bytes16", "bytes20", "bytes32",
    "memory", "storage", "calldata", "indexed",
    "constructor", "fallback", "receive",
    "var", "wei", "szabo", "finney", "ether",
    "seconds", "minutes", "hours", "days", "weeks", "years",
    "transfer", "send", "call", "delegatecall", "staticcall",
    "keccak256", "sha256", "sha3", "ripemd160", "ecrecover",
    "addmod", "mulmod",
    "push", "pop", "length",
    "balance", "code", "codehash",
    "origin", "sender", "value", "data", "sig", "gas", "gasprice",
    "coinbase", "timestamp", "number", "difficulty", "gaslimit", "chainid",
    "log0", "log1", "log2", "log3", "log4",
    "assembly", "let", "switch", "case", "default",
    "import", "from", "as",
    "type", "error", "unchecked",
    "ERC20", "IERC20", "SafeMath", "Ownable", "Context",
    "Approval", "Transfer",
}

# Well-known identifiers from OpenZeppelin / common patterns
COMMON_IDENTIFIERS = {
    "totalSupply", "balanceOf", "allowance", "approve", "transferFrom",
    "safeTransferFrom", "safeTransfer", "mint", "burn",
    "owner", "onlyOwner", "renounceOwnership", "transferOwnership",
    "add", "sub", "mul", "div", "mod",  # SafeMath
}


def strip_comments(source):
    """Remove // and /* */ comments, preserving line count."""
    # Remove block comments, replacing with same number of newlines
    def block_repl(m):
        return '\n' * m.group(0).count('\n')
    source = re.sub(r'/\*.*?\*/', block_repl, source, flags=re.DOTALL)
    # Remove line comments
    lines = source.split('\n')
    out = []
    for line in lines:
        # Don't strip inside string literals (simplistic)
        in_str = False
        result = []
        i = 0
        while i < len(line):
            if line[i] in ('"', "'") and not in_str:
                in_str = line[i]
                result.append(line[i])
            elif line[i] == in_str:
                in_str = False
                result.append(line[i])
            elif line[i:i+2] == '//' and not in_str:
                break
            else:
                result.append(line[i])
            i += 1
        out.append(''.join(result).rstrip())
    return '\n'.join(out)


def rename_identifiers(source):
    """Rename user-defined identifiers to generic names."""
    # Collect all identifiers
    tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', source)
    unique = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Build rename map: skip keywords and common identifiers
    rename_map = {}
    import random
    import string
    
    def generate_random_id(length=6):
        # Start with a letter
        first = random.choice(string.ascii_lowercase)
        # Follow with letters or numbers
        rest = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length - 1))
        return first + rest

    used_ids = set()
    for ident in unique:
        if ident in SOLIDITY_KEYWORDS or ident in COMMON_IDENTIFIERS:
            continue
        if ident.startswith("0x"):  # hex literals
            continue
        if len(ident) <= 1:  # single chars like i, j, x
            continue
            
        # Generate a unique random identifier
        while True:
            new_id = generate_random_id(random.randint(5, 8))
            if new_id not in used_ids and new_id not in SOLIDITY_KEYWORDS and new_id not in COMMON_IDENTIFIERS:
                used_ids.add(new_id)
                rename_map[ident] = new_id
                break

    # Apply renames (longest first to avoid partial replacements)
    for old, new in sorted(rename_map.items(), key=lambda x: -len(x[0])):
        source = re.sub(r'\b' + re.escape(old) + r'\b', new, source)

    return source


def obfuscate(source):
    """Full obfuscation pipeline."""
    result = strip_comments(source)
    result = rename_identifiers(result)
    return result


def main():
    print("=== Tier A: Obfuscating SBCurated Contracts ===")
    os.makedirs(OUT_SOL_DIR, exist_ok=True)

    # Load ground truth
    gt = {}
    with open(GT_FILE, 'r') as f:
        for line in f:
            d = json.loads(line)
            gt[d["filename"]] = d

    success = 0
    failed = 0
    labels_out = []
    annos_out = []

    sol_files = sorted([f for f in os.listdir(SOL_DIR) if f.endswith('.sol')])
    print(f"Found {len(sol_files)} .sol files")

    for fn in sol_files:
        filepath = os.path.join(SOL_DIR, fn)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()

            obfuscated = obfuscate(source)

            # Verify line count preserved
            orig_lines = len(source.split('\n'))
            obf_lines = len(obfuscated.split('\n'))
            assert orig_lines == obf_lines, f"Line count mismatch: {orig_lines} vs {obf_lines}"

            # Save obfuscated file
            out_path = os.path.join(OUT_SOL_DIR, fn)
            with open(out_path, 'w') as f:
                f.write(obfuscated)

            # Create label entry
            label_entry = {
                "dataset": "SBCurated_Obfuscated",
                "filename": fn,
                "source_code": obfuscated,
                "labels": gt[fn]["vulnerability_types"] if fn in gt else [],
            }
            labels_out.append(label_entry)

            # Transfer annotations (line numbers are preserved)
            if fn in gt and gt[fn]["vulnerable_lines"]:
                anno_entry = {
                    "dataset": "SBCurated_Obfuscated",
                    "filename": fn,
                    "vulnerability_types": gt[fn]["vulnerability_types"],
                    "vulnerable_lines": gt[fn]["vulnerable_lines"],
                }
                annos_out.append(anno_entry)

            success += 1
        except Exception as e:
            print(f"  FAIL {fn}: {e}")
            failed += 1

    # Save outputs
    with open(OUT_LABELS, 'w') as f:
        for entry in labels_out:
            f.write(json.dumps(entry) + '\n')

    with open(OUT_ANNOS, 'w') as f:
        for entry in annos_out:
            f.write(json.dumps(entry) + '\n')

    print(f"\nObfuscation complete: {success} success, {failed} failed")
    print(f"Labels: {OUT_LABELS}")
    print(f"Annotations: {OUT_ANNOS}")
    print(f"Sol files: {OUT_SOL_DIR}/")


if __name__ == "__main__":
    main()
