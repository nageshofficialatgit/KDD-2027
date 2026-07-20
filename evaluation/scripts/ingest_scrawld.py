import json
import os
import re

SCRAWLD_DIR = "./Dataset_for_first/ScrawlD"
OUTPUT_DIR = "./evaluation/registry"

LABEL_MAP = {
    "RENT": "Reentrancy",
    "ARTHM": "Integer_Overflow_Underflow",
    "TimeM": "Timestamp_Dependence",
    "TimeO": "Transaction_Ordering_Dependence_Front_Running",
    "Tx-Origin": "tx_origin",
    "UE": "Unchecked_Low_Level_Calls"
}

def load_function_lines():
    func_map = {}
    path = os.path.join(SCRAWLD_DIR, "data", "function_lines.txt")
    with open(path, 'r') as f:
        for line in f:
             parts = line.strip().split(';')
             if len(parts) == 4:
                 fname, func, start, end = parts
                 fname = os.path.basename(fname)
                 if fname not in func_map:
                     func_map[fname] = {}
                 func_map[fname][func] = (int(start), int(end))
    return func_map

def ingest():
    print("--- Ingesting ScrawlD for Phase 2 ---")
    func_map = load_function_lines()
    
    with open(os.path.join(SCRAWLD_DIR, "data", "majority_result.json"), 'r') as f:
        majority = json.load(f)
        
    merged_labels = []
    merged_annotations = []
    
    for filename, vulns in majority.items():
        # ScrawlD filenames often have _ext.sol or are paths
        clean_name = os.path.basename(filename)
        src_path = os.path.join(SCRAWLD_DIR, "contract_sources", clean_name)
        
        if not os.path.exists(src_path):
            continue
            
        with open(src_path, 'r') as f:
            source_code = f.read()
            
        # 1. Create label entry
        label_entry = {
            "dataset": "ScrawlD",
            "filename": clean_name,
            "source_code": source_code,
            "labels": {v: 0 for v in ["Reentrancy", "Integer_Overflow_Underflow", "Access_Control", "Timestamp_Dependence", "Transaction_Ordering_Dependence_Front_Running", "tx_origin", "Bad_Randomness", "Unchecked_Low_Level_Calls"]}
        }
        
        # 2. Extract annotations
        vulnerable_lines = set()
        for vtype, indicators in vulns.items():
            mapped_type = LABEL_MAP.get(vtype)
            if not mapped_type:
                continue
                
            label_entry["labels"][mapped_type] = 1
            
            for ind in indicators:
                if isinstance(ind, int):
                    vulnerable_lines.add(ind)
                elif isinstance(ind, str) and ind != "Yes":
                    # Function name (Reentrancy)
                    if clean_name in func_map and ind in func_map[clean_name]:
                        start, end = func_map[clean_name][ind]
                        for i in range(start, end + 1):
                            vulnerable_lines.add(i)
                            
        merged_labels.append(label_entry)
        if vulnerable_lines:
            merged_annotations.append({
                "dataset": "ScrawlD",
                "filename": clean_name,
                "vulnerable_lines": sorted(list(vulnerable_lines))
            })

    # Save
    with open(os.path.join(OUTPUT_DIR, "scrawld_labels.jsonl"), 'w') as f:
        for entry in merged_labels:
            f.write(json.dumps(entry) + '\n')
            
    with open(os.path.join(OUTPUT_DIR, "scrawld_annotations.jsonl"), 'w') as f:
        for entry in merged_annotations:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Ingested {len(merged_labels)} contracts and {len(merged_annotations)} annotations.")

if __name__ == "__main__":
    ingest()
