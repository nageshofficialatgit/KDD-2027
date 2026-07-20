import os
import json
import re
import hashlib

# Paths
UNIFIED_LABELS = "./evaluation/data/unified_labels.jsonl"
SCRAWLD_VULN_JSON = "./Dataset_for_first/ScrawlD/data/vulnerabilities.json"
OUTPUT_FILE = "./evaluation/data/line_annotations.jsonl"

# Vuln Map for ScrawlD
# Note: These are rough heuristics. Ideally verified against ScrawlD paper.
# ARTHM -> Integer_Overflow_Underflow
# RENT -> Reentrancy
# UE -> Unchecked_Low_Level_Calls
# TimeM -> Timestamp_Dependence (Most tools label it so)
# TimeO -> Transaction_Ordering_Dependence_Front_Running (TOD)
# DOS -> (Often related to TOD or Access Control, but we'll map if clear)
SCRAWLD_MAP = {
    "ARTHM": "Integer_Overflow_Underflow",
    "RENT": "Reentrancy",
    "UE": "Unchecked_Low_Level_Calls",
    "TimeM": "Timestamp_Dependence",
    "TimeO": "Transaction_Ordering_Dependence_Front_Running"
}

def compute_code_hash(content):
    content_no_comments = re.sub(r'//.*', '', content)
    content_no_whitespace = re.sub(r'\s+', '', content_no_comments)
    return hashlib.sha256(content_no_whitespace.encode('utf-8')).hexdigest()

def extract_sbcurated_lines(source):
    # Matches patterns like "* @vulnerable_at_lines: 23,25,33"
    matches = re.findall(r'@vulnerable_at_lines:\s*([\d,\s]+)', source)
    lines = set()
    for m in matches:
        for part in m.split(','):
            if part.strip().isdigit():
                lines.add(int(part.strip()))
    return sorted(list(lines))

def main():
    print("=== Step 0c: Line-Level Annotation Parser ===")
    
    # Load ScrawlD Vuln Data
    with open(SCRAWLD_VULN_JSON, 'r') as f:
        scrawld_data = json.load(f)
    print(f"Loaded ScrawlD vulnerability data for {len(scrawld_data)} entries.")

    line_docs = 0
    with open(UNIFIED_LABELS, 'r') as f, open(OUTPUT_FILE, 'w') as out:
        for line in f:
            item = json.loads(line)
            dataset = item['dataset']
            filename = item['filename']
            src = item['source_code']
            v_lines = []
            
            if dataset == "SBCurated":
                # SBCurated has annotations in comments
                v_lines = extract_sbcurated_lines(src)
            elif dataset == "ScrawlD" or (dataset == "SBWILD" and filename in scrawld_data):
                # We might use ScrawlD logic for matched SBWILD too if applicable, 
                # but ScrawlD labels are preferred if present.
                key = filename if filename in scrawld_data else (filename + "_ext.sol" if filename + "_ext.sol" in scrawld_data else None)
                if key:
                    entry = scrawld_data[key]
                    all_lines = set()
                    for cat, tools in entry.items():
                        if cat in SCRAWLD_MAP:
                            # It's one of our target vulns
                            for tool, t_lines in tools.items():
                                if isinstance(t_lines, list):
                                    for line_ref in t_lines:
                                        if isinstance(line_ref, int):
                                            all_lines.add(line_ref)
                    v_lines = sorted(list(all_lines))

            # Only write if we found line numbers
            if v_lines:
                out.write(json.dumps({
                    "dataset": dataset,
                    "filename": filename,
                    "stripped_hash": item['stripped_hash'],
                    "vulnerable_lines": v_lines
                }) + "\n")
                line_docs += 1
                
    print(f"Extraction complete. Found line annotations for {line_docs} contracts.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
