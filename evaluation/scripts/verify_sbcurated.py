import os
import re
import json
import pandas as pd

# Paths
DATASET_DIR = "./Dataset SmartContracts/SBCurated_141/SBCurated_141/SolidityFiles_Data"
CSV_PATH = "./Dataset SmartContracts/SBCurated_141/SBCurated_141/Labelling/labelled_sbcurated (1).csv"
OUTPUT_PATH = "./evaluation/data/verified_sbcurated_ground_truth.jsonl"

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
    print("=== Verifying SBCurated Ground Truth ===")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    
    # Identify vulnerability columns
    # We skip the first column if it's an index, and the 'filename' column
    # The columns are: filename,Reentrancy,Integer_Overflow_Underflow,...
    vuln_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'filename', '0']]
    print(f"Vulnerability categories found in CSV: {vuln_columns}")
    
    verified_data = []
    
    for index, row in df.iterrows():
        filename = row['filename']
        file_path = os.path.join(DATASET_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
            
        # Extract lines
        vuln_lines = extract_sbcurated_lines(source)
        
        # Extract labels from CSV
        present_vulns = [vuln for vuln in vuln_columns if row[vuln] == 1]
        
        verified_data.append({
            "dataset": "SBCurated",
            "filename": filename,
            "vulnerability_types": present_vulns,
            "vulnerable_lines": vuln_lines
        })
        
    # Write to JSONL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for item in verified_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nSuccessfully verified {len(verified_data)} SBCurated contracts.")
    print(f"Saved verified ground truth to: {OUTPUT_PATH}")

    # Let's print a couple of examples for the user
    print("\nSample Output:")
    for i in range(min(3, len(verified_data))):
        print(json.dumps(verified_data[i], indent=2))

if __name__ == "__main__":
    main()
