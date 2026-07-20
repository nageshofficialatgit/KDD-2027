import os
import json
import hashlib
import re

# Paths
BASE_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
SCRAWLD_VULN_JSON = os.path.join(BASE_DIR, "Dataset_for_first/ScrawlD/data/vulnerabilities.json")
LOCAL_MATCHED_JSON = os.path.join(BASE_DIR, "Implementation/evaluation/data/scrawld_matched_local.json")
DOWNLOADED_DIR = os.path.join(BASE_DIR, "Implementation/evaluation/data/scrawld_sources")
UNIFIED_LABELS = os.path.join(BASE_DIR, "Implementation/evaluation/data/unified_labels.jsonl")

ALL_VULN_TYPES = [
    "Reentrancy", "Integer_Overflow_Underflow", "Access_Control", 
    "Timestamp_Dependence", "Transaction_Ordering_Dependence_Front_Running", 
    "tx_origin", "Bad_Randomness", "Unchecked_Low_Level_Calls"
]

SCRAWLD_MAP = {
    "RENT": "Reentrancy",
    "ARTHM": "Integer_Overflow_Underflow",
    "UE": "Unchecked_Low_Level_Calls",
    "TimeM": "Timestamp_Dependence",
    "TimeO": "Transaction_Ordering_Dependence_Front_Running"
}

def compute_code_hash(content):
    content_no_comments = re.sub(r'//.*', '', content)
    content_no_whitespace = re.sub(r'\s+', '', content_no_comments)
    return hashlib.sha256(content_no_whitespace.encode('utf-8')).hexdigest()

def main():
    print("=== Merging ScrawlD into Unified Labels ===")
    
    # Load ScrawlD Vuln Data
    with open(SCRAWLD_VULN_JSON, 'r') as f:
        scrawld_vulns = json.load(f)
    
    # Load Local Matches
    if os.path.exists(LOCAL_MATCHED_JSON):
        with open(LOCAL_MATCHED_JSON, 'r') as f:
            local_matches = json.load(f)
    else:
        local_matches = {}

    processed = 0
    
    # Get set of already matched/processed addresses to avoid duplicates
    processed_files = set()
    if os.path.exists(UNIFIED_LABELS):
        with open(UNIFIED_LABELS, 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['dataset'] == "ScrawlD":
                    processed_files.add(item['filename'])

    with open(UNIFIED_LABELS, 'a', encoding='utf-8') as outfile:
        # Case 1: Local Matches
        for name, local_path in local_matches.items():
            # name is typically 0x..._ext.sol
            addr = name.replace('_ext.sol', '').replace('.sol', '')
            if addr in processed_files: continue
            
            key = name if name in scrawld_vulns else (addr + "_ext.sol")
            if key not in scrawld_vulns:
                # Try just addr
                if addr in scrawld_vulns: key = addr
                else: continue
            
            vuln_entry = scrawld_vulns[key]
            labels = {v: 0 for v in ALL_VULN_TYPES}
            for cat, tools in vuln_entry.items():
                if cat in SCRAWLD_MAP:
                    labels[SCRAWLD_MAP[cat]] = 1
            
            if not os.path.exists(local_path): continue
            try:
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    src = f.read()
            except: continue
            
            outfile.write(json.dumps({
                "dataset": "ScrawlD", "filename": addr,
                "stripped_hash": compute_code_hash(src),
                "source_code": src, "labels": labels
            }) + "\n")
            processed += 1
            processed_files.add(addr)

        # Case 2: Downloaded ScrawlD sources
        if os.path.exists(DOWNLOADED_DIR):
            for fname in os.listdir(DOWNLOADED_DIR):
                if not fname.endswith('.sol'): continue
                addr = fname.replace('.sol', '')
                if addr in processed_files: continue
                
                key = addr if addr in scrawld_vulns else (addr + "_ext.sol")
                if key not in scrawld_vulns: continue
                
                vuln_entry = scrawld_vulns[key]
                labels = {v: 0 for v in ALL_VULN_TYPES}
                for cat, tools in vuln_entry.items():
                    if cat in SCRAWLD_MAP:
                        labels[SCRAWLD_MAP[cat]] = 1
                
                src_path = os.path.join(DOWNLOADED_DIR, fname)
                with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
                    src = f.read()
                
                outfile.write(json.dumps({
                    "dataset": "ScrawlD", "filename": addr,
                    "stripped_hash": compute_code_hash(src),
                    "source_code": src, "labels": labels
                }) + "\n")
                processed += 1
                processed_files.add(addr)

    print(f"✅ Appended {processed} ScrawlD entries to unified_labels.jsonl.")

if __name__ == "__main__":
    main()
