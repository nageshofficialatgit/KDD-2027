#!/usr/bin/env python3
"""
Dataset Cleaner + Version Bucketer
-----------------------------------
Streams `merged_train.json` (18 GB JSON array) via ijson,
cleans each contract, and writes per-version-bucket JSONL files.

Output: Dataset_for_first/cleaned_buckets/v{major}.{minor}.jsonl

Usage:
    python clean_dataset.py [--input PATH] [--output-dir DIR]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import ijson
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import sibling module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solc_resolver import detect_pragma_version, version_to_bucket  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = str(PROJECT_ROOT / "Dataset_for_first" / "merged_train.json")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "Dataset_for_first" / "cleaned_buckets")


# ---------------------------------------------------------------------------
# Source Cleaning  (standalone — no legacy dependency)
# ---------------------------------------------------------------------------

_PRAGMA_LINE_RE = re.compile(r'^\s*pragma\b', re.MULTILINE)
_IMPORT_LINE_RE = re.compile(r'^\s*import\b[^;]*;', re.MULTILINE)
_UNCHECKED_AFTER_SEMI = re.compile(r'([;])\s*unchecked\s*\{(.*?)\}', re.DOTALL)
_UNCHECKED_AFTER_BRACE = re.compile(r'([\}])\s*unchecked\s*\{(.*?)\}', re.DOTALL)


def clean_source(source: str) -> Optional[str]:
    """
    Clean a Solidity source string:
      1. Normalize line endings
      2. Keep only the first pragma line
      3. Remove all import lines
      4. Fix floating `unchecked {}` blocks
      5. Return None if result is trivially empty
    """
    if not source or not source.strip():
        return None

    # Normalize \r\n → \n
    source = source.replace('\r\n', '\n').replace('\r', '\n')

    lines = source.split('\n')
    cleaned: list[str] = []
    pragma_found = False
    first_valid = False

    for line in lines:
        stripped = line.strip()

        # Skip junk before first valid Solidity line
        if not first_valid:
            if (stripped.startswith('//')
                    or stripped.startswith('/*')
                    or stripped.startswith('pragma')
                    or stripped.startswith('contract')
                    or stripped.startswith('interface')
                    or stripped.startswith('library')
                    or stripped.startswith('abstract')
                    or stripped.startswith('// SPDX')):
                first_valid = True
            else:
                continue

        # Pragma handling: keep only the first
        if stripped.startswith('pragma'):
            if not pragma_found:
                cleaned.append(line)
                pragma_found = True
            continue

        # Remove imports
        if stripped.startswith('import'):
            continue

        cleaned.append(line)

    if not cleaned:
        return None

    content = '\n'.join(cleaned)

    # Fix unchecked blocks
    content = _UNCHECKED_AFTER_SEMI.sub(r'\1\2', content)
    content = _UNCHECKED_AFTER_BRACE.sub(r'\1\2', content)

    # Final check: must contain at least one meaningful definition
    if not any(kw in content for kw in ('contract ', 'library ', 'interface ')):
        return None

    return content


# ---------------------------------------------------------------------------
# Streaming processor
# ---------------------------------------------------------------------------

def stream_and_clean(
    input_path: str,
    output_dir: str,
    *,
    max_records: Optional[int] = None,
) -> dict[str, int]:
    """
    Stream-parse the JSON array in *input_path*, clean each record,
    and write to per-version-bucket JSONL files under *output_dir*.

    Returns a dict {bucket_key: count} summarising output.
    """
    os.makedirs(output_dir, exist_ok=True)

    bucket_handles: dict[str, 'TextIO'] = {}
    bucket_counts: dict[str, int] = {}
    total = 0
    skipped = 0

    try:
        with open(input_path, 'rb') as f:
            items = ijson.items(f, 'item')
            pbar = tqdm(items, desc="Cleaning contracts", unit="rec")

            for item in pbar:
                if max_records is not None and total >= max_records:
                    break
                total += 1

                # ----- Filter -----
                source = item.get("source_code") or ""
                if not source.strip():
                    skipped += 1
                    continue

                lang = item.get("language", "")
                if lang and lang.lower() != "solidity":
                    skipped += 1
                    continue

                # ----- Clean -----
                cleaned = clean_source(source)
                if cleaned is None:
                    skipped += 1
                    continue

                # ----- Detect version & bucket -----
                version = detect_pragma_version(cleaned)
                if version is None:
                    skipped += 1
                    continue

                bucket = version_to_bucket(version)

                # ----- Write -----
                record = {
                    "contract_name": item.get("contract_name", ""),
                    "contract_address": item.get("contract_address", ""),
                    "source_code": cleaned,
                    "compiler_version": version,
                    "original_compiler": item.get("compiler_version", ""),
                }
                line = json.dumps(record, ensure_ascii=False)

                if bucket not in bucket_handles:
                    path = os.path.join(output_dir, f"v{bucket}.jsonl")
                    bucket_handles[bucket] = open(path, 'w', encoding='utf-8')
                    bucket_counts[bucket] = 0

                bucket_handles[bucket].write(line + '\n')
                bucket_counts[bucket] += 1

                if total % 10000 == 0:
                    pbar.set_postfix(buckets=len(bucket_counts), skipped=skipped)

    finally:
        for fh in bucket_handles.values():
            fh.close()

    print(f"\n--- Cleaning Complete ---")
    print(f"Total scanned:  {total}")
    print(f"Skipped:        {skipped}")
    print(f"Written:        {total - skipped}")
    print(f"Buckets:        {len(bucket_counts)}")
    for bk in sorted(bucket_counts, key=lambda k: bucket_counts[k], reverse=True):
        print(f"  v{bk}: {bucket_counts[bk]:,}")

    return bucket_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean & bucket Solidity dataset")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to merged_train.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Dir for per-version JSONL output")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit records processed (for testing)")
    args = parser.parse_args()

    stream_and_clean(args.input, args.output_dir, max_records=args.max_records)


if __name__ == "__main__":
    main()
