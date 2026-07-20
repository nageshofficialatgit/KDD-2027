import os
import json
import hashlib
import re
from collections import defaultdict

# Paths
UNIFIED_LABELS = "./evaluation/data/unified_labels.jsonl"
OUTPUT_REPORT = "./evaluation/data/matching_report.json"

def main():
    print("=== Step 0d: Source deduplicator & Registry Analysis ===")
    
    hash_to_entries = defaultdict(list)
    dataset_counts = defaultdict(int)
    total_entries = 0

    with open(UNIFIED_LABELS, 'r') as f:
        for line in f:
            item = json.loads(line)
            h = item['stripped_hash']
            hash_to_entries[h].append({
                "dataset": item['dataset'],
                "filename": item['filename']
            })
            dataset_counts[item['dataset']] += 1
            total_entries += 1

    # Analysis
    unique_contracts = len(hash_to_entries)
    duplicate_groups = {h: entries for h, entries in hash_to_entries.items() if len(entries) > 1}
    cross_dataset_duplicates = []
    
    for h, entries in duplicate_groups.items():
        datasets = set(e['dataset'] for e in entries)
        if len(datasets) > 1:
            cross_dataset_duplicates.append({
                "hash": h,
                "occurrences": entries
            })

    report = {
        "summary": {
            "total_labeled_entries": total_entries,
            "unique_source_contracts": unique_contracts,
            "duplicate_count": total_entries - unique_contracts,
            "cross_dataset_overlaps": len(cross_dataset_duplicates)
        },
        "dataset_stats": dataset_counts,
        "cross_dataset_examples": cross_dataset_duplicates[:20]  # Just top 20 for preview
    }

    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Registry Analysis Complete.")
    print(f"Total Entries: {total_entries}")
    print(f"Unique Contracts: {unique_contracts}")
    print(f"Duplicates: {total_entries - unique_contracts}")
    print(f"Cross-dataset Overlaps: {len(cross_dataset_duplicates)}")
    print(f"Report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
