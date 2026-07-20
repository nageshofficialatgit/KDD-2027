import os
import json
import csv
import hashlib
import re

BASE_DIR = "./Dataset SmartContracts"
OUTPUT_FILE = "./evaluation/data/unified_labels.jsonl"

ALL_VULN_TYPES = [
    "Reentrancy", 
    "Integer_Overflow_Underflow", 
    "Access_Control", 
    "Timestamp_Dependence", 
    "Transaction_Ordering_Dependence_Front_Running", 
    "tx_origin", 
    "Bad_Randomness",
    "Unchecked_Low_Level_Calls"
]

def compute_code_hash(content):
    content_no_comments = re.sub(r'//.*', '', content)
    content_no_whitespace = re.sub(r'\s+', '', content_no_comments)
    return hashlib.sha256(content_no_whitespace.encode('utf-8')).hexdigest()

def merge_standard(outfile, name, csv_rel, src_rel, filename_col):
    csv_path = os.path.join(BASE_DIR, csv_rel)
    src_dir = os.path.join(BASE_DIR, src_rel)
    processed = 0
    missing = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get(filename_col)
            if not filename: continue
            
            labels = {v: (1 if str(row.get(v, "0")).strip() == "1" else 0) for v in ALL_VULN_TYPES}
            
            src_path = os.path.join(src_dir, filename)
            if not os.path.exists(src_path):
                if not filename.endswith('.sol') and os.path.exists(src_path + '.sol'):
                    src_path += '.sol'
                else:
                    missing += 1
                    continue
            
            with open(src_path, 'r', encoding='utf-8', errors='ignore') as sf:
                src = sf.read()
            
            outfile.write(json.dumps({
                "dataset": name, "filename": filename, 
                "stripped_hash": compute_code_hash(src), 
                "source_code": src, "labels": labels
            }) + "\n")
            processed += 1
    return processed, missing

def merge_sbcurated(outfile):
    # Special handling for SBCurated: index column offset
    csv_path = os.path.join(BASE_DIR, "SBCurated_141/SBCurated_141/Labelling/labelled_sbcurated (1).csv")
    src_dir = os.path.join(BASE_DIR, "SBCurated_141/SBCurated_141/SolidityFiles_Data")
    processed = 0
    missing = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Header: filename, Reentrancy (1), Integer (2), Access (3), Timestamp (4), TOD (5), tx_origin (6), Unchecked (7)
        # Data: index, filename (1), Reentrancy (2), ...
        for row in reader:
            if len(row) < 2: continue
            filename = row[1]
            labels = {
                "Reentrancy": 1 if row[2] == "1" else 0,
                "Integer_Overflow_Underflow": 1 if row[3] == "1" else 0,
                "Access_Control": 1 if row[4] == "1" else 0,
                "Timestamp_Dependence": 1 if row[5] == "1" else 0,
                "Transaction_Ordering_Dependence_Front_Running": 1 if row[6] == "1" else 0,
                "tx_origin": 1 if row[7] == "1" else 0,
                "Unchecked_Low_Level_Calls": 1 if row[8] == "1" else 0,
                "Bad_Randomness": 0
            }
            
            src_path = os.path.join(src_dir, filename)
            if not os.path.exists(src_path):
                missing += 1
                continue
            
            with open(src_path, 'r', encoding='utf-8', errors='ignore') as sf:
                src = sf.read()
            
            outfile.write(json.dumps({
                "dataset": "SBCurated", "filename": filename,
                "stripped_hash": compute_code_hash(src),
                "source_code": src, "labels": labels
            }) + "\n")
            processed += 1
    return processed, missing

def merge_codesmells(outfile):
    # Special handling for CodeSmells: header starts with a comma
    csv_path = os.path.join(BASE_DIR, "labelled_codesmell (1).csv")
    src_dir = os.path.join(BASE_DIR, "CodeSmell_OriginalFiles")
    processed = 0
    missing = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("basename")
            if not filename: continue
            
            labels = {v: (1 if str(row.get(v, "0")).strip() == "1" else 0) for v in ALL_VULN_TYPES}
            
            # The dir might be in 'CodeSmell_OriginalFiles/CodeSmell_OriginalFiles' if it was a nested zip
            src_path = os.path.join(src_dir, filename)
            if not os.path.exists(src_path):
                # Check nested
                alt_dir = os.path.join(src_dir, "CodeSmell_OriginalFiles")
                if os.path.exists(os.path.join(alt_dir, filename)):
                    src_path = os.path.join(alt_dir, filename)
                else:
                    missing += 1
                    continue
            
            with open(src_path, 'r', encoding='utf-8', errors='ignore') as sf:
                src = sf.read()
                
            outfile.write(json.dumps({
                "dataset": "CodeSmells", "filename": filename,
                "stripped_hash": compute_code_hash(src),
                "source_code": src, "labels": labels
            }) + "\n")
            processed += 1
    return processed, missing

def main():
    print("=== Step 0a: Label Merger (Refined) ===")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    total_p = 0
    total_m = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Standard ones
        for name, csv_rel, src_rel, col in [
            ("ESC", "ESC_final/ESC_final/ESC_labelling/labelled_ESC_1001.csv", "ESC_final/ESC_final/ESC_1001_dataset", "filename"),
            ("ICSE", "ICSE_final/ICSE_final/ICSE_Labelling/labelled_ICSE.csv", "ICSE_final/ICSE_final/Dataset_ICSE", "filename"),
            ("SBWILD", "SBWILD/SBWILD/Labelling/laballed_sbw.csv", "SBWILD/SBWILD/SolidityFiles_Data", "filename")
        ]:
            p, m = merge_standard(outfile, name, csv_rel, src_rel, col)
            print(f"✅ [{name}] Merged {p} entries. (Missing: {m})")
            total_p += p; total_m += m
            
        p, m = merge_sbcurated(outfile)
        print(f"✅ [SBCurated] Merged {p} entries. (Missing: {m})")
        total_p += p; total_m += m
        
        p, m = merge_codesmells(outfile)
        print(f"✅ [CodeSmells] Merged {p} entries. (Missing: {m})")
        total_p += p; total_m += m

    print(f"\n--- Label Merging Complete ---")
    print(f"Total entries unified: {total_p}")
    print(f"Missing source code instances: {total_m}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
