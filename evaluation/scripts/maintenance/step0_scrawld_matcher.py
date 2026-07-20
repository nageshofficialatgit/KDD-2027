import os
import json
import hashlib
from glob import glob
from collections import defaultdict
import re

SCRAWLD_JSON = "./Dataset_for_first/ScrawlD/data/vulnerabilities.json"
DATADUMP_DIR = "./Dataset SmartContracts/datadump_cleaned"
OUTPUT_DIR = "./evaluation/data"

def get_clean_address(filename):
    """Extracts base hex address from various filename formats"""
    # Handles `0x123_ext.sol`, `0x123.sol`, etc.
    match = re.search(r'(0x[a-fA-F0-9]{40})', filename)
    if match:
        return match.group(1).lower()
    return None

def compute_code_hash(filepath):
    """Computes SHA-256 of code after stripping whitespace/comments for structural deduplication"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Strip simple single-line comments and all whitespace for a sturdy functional hash
        # (This is a simplified hash just for locating exact duplicates)
        content_no_comments = re.sub(r'//.*', '', content)
        content_no_whitespace = re.sub(r'\s+', '', content_no_comments)
        
        return hashlib.sha256(content_no_whitespace.encode('utf-8')).hexdigest()
    except Exception as e:
        return None

def main():
    print("=== Step 0_scrawl: ScrawlD Matching & Deduplication ===")
    
    # 1. Load ScrawlD Target Addresses
    with open(SCRAWLD_JSON, 'r') as f:
        scrawld_data = json.load(f)
    
    scrawld_filenames = list(scrawld_data.keys())
    scrawld_addresses = {get_clean_address(f): f for f in scrawld_filenames if get_clean_address(f)}
    
    print(f"Loaded {len(scrawld_filenames)} entries from ScrawlD.")
    print(f"Extracted {len(scrawld_addresses)} valid hex addresses.")
    
    # 2. Load Local Datadump files
    local_files = glob(os.path.join(DATADUMP_DIR, "*.sol"))
    print(f"\nFound {len(local_files)} local `.sol` files in `datadump_cleaned`.")
    
    local_addresses = {}
    for lf in local_files:
        addr = get_clean_address(os.path.basename(lf))
        if addr:
            local_addresses[addr] = lf
            
    # 3. Exactly Match by Address
    matched_by_address = {}
    missing_addresses = []
    
    for scrawld_addr, orig_filename in scrawld_addresses.items():
        if scrawld_addr in local_addresses:
            matched_by_address[orig_filename] = local_addresses[scrawld_addr]
        else:
            missing_addresses.append(orig_filename)
            
    print(f"\n--- Address Match Results ---")
    print(f"✅ Exact Local Matches: {len(matched_by_address)}")
    print(f"❌ Missing Sources: {len(missing_addresses)}")
    
    # 4. Hash Deduplication for matched files
    print("\nComputing SHA-256 hashes for deduplication of matched active files...")
    hash_to_files = defaultdict(list)
    for orig_file, local_path in matched_by_address.items():
        h = compute_code_hash(local_path)
        if h:
            hash_to_files[h].append(orig_file)
            
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    print(f"Found {len(duplicates)} hash-collision clusters within our local ScrawlD matches.")
    
    # Save reports
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, 'scrawld_matched_local.json'), 'w') as f:
        json.dump(matched_by_address, f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, 'scrawld_missing_sources.json'), 'w') as f:
        json.dump(missing_addresses, f, indent=2)
        
    print(f"\nSaved matching reports to {OUTPUT_DIR}/")
    print("Next step: Download missing sources from Etherscan or proceed with the matched subset.")

if __name__ == "__main__":
    main()
