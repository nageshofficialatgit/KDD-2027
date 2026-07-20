import os
import json
import re
from tree_sitter import Language, Parser
import tree_sitter_solidity

# Paths
LABELS_PATH = "./evaluation/registry/labels.jsonl"
OUT_DIR = "./diff_based_detection/data"
OUT_PATH = os.path.join(OUT_DIR, "SBCurated_Diffs.jsonl")

def setup_parser():
    SOLIDITY_LANGUAGE = Language(tree_sitter_solidity.language())
    parser = Parser()
    parser.language = SOLIDITY_LANGUAGE
    return parser

def extract_vuln_lines(source_code):
    match = re.search(r'@vulnerable_at_lines:\s*([0-9,]+)', source_code)
    if match:
        lines = match.group(1).split(',')
        return [int(l) for l in lines]
    return []

def find_enclosing_function(node, start_byte, end_byte):
    if node.start_byte <= start_byte and node.end_byte >= end_byte:
        if node.type == 'function_definition':
            return node
        for child in node.children:
            res = find_enclosing_function(child, start_byte, end_byte)
            if res:
                return res
    return None

def find_fallback_block(node, start_byte, end_byte):
    if node.start_byte <= start_byte and node.end_byte >= end_byte:
        if node.type == 'block':
            return node
        for child in node.children:
            res = find_fallback_block(child, start_byte, end_byte)
            if res:
                return res
    return node

def process_file(parser, record):
    source_code = record["source_code"]
    source_bytes = bytes(source_code, "utf8")
    
    vuln_lines = extract_vuln_lines(source_code)
    
    tree = parser.parse(source_bytes)
    root = tree.root_node
    
    diff_chunk = ""
    diff_start_line = 0
    diff_end_line = 0
    
    if vuln_lines:
        vuln_line = vuln_lines[0] - 1 # 0-indexed
        
        lines = source_code.split("\n")
        if vuln_line < len(lines):
            target_start = sum(len(l) + 1 for l in lines[:vuln_line])
            target_end = target_start + len(lines[vuln_line])
            
            enclosing = find_enclosing_function(root, target_start, target_end)
            if not enclosing:
                enclosing = find_fallback_block(root, target_start, target_end)
                
            if enclosing:
                diff_chunk = source_bytes[enclosing.start_byte:enclosing.end_byte].decode('utf8', errors='ignore')
                diff_start_line = enclosing.start_point[0]
                diff_end_line = enclosing.end_point[0]
    else:
        def get_first_func(n):
            if n.type == 'function_definition':
                return n
            for c in n.children:
                res = get_first_func(c)
                if res: return res
            return None
        
        f = get_first_func(root)
        if f:
            diff_chunk = source_bytes[f.start_byte:f.end_byte].decode('utf8', errors='ignore')
            diff_start_line = f.start_point[0]
            diff_end_line = f.end_point[0]
            
    mock_diff = f"@@ -{diff_start_line}, +{diff_start_line} @@\n" + diff_chunk
    
    record["diff_chunk"] = mock_diff
    record["diff_start_line"] = diff_start_line
    record["diff_end_line"] = diff_end_line
    return record

def main():
    print("Starting...")
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Setting up parser...")
    parser = setup_parser()
    print("Parser setup complete.")
    
    count = 0
    print(f"Reading from {LABELS_PATH}")
    
    # Try just loading a few lines to test if IO is hanging
    with open(LABELS_PATH, 'r') as f_in, open(OUT_PATH, 'w') as f_out:
        for line in f_in:
            if not line.strip(): continue
            record = json.loads(line)
            if record.get("dataset") == "SBCurated":
                diff_record = process_file(parser, record)
                f_out.write(json.dumps(diff_record) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} SBCurated files...")
            
    print(f"Processed {count} total SBCurated files. Diff dataset saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
