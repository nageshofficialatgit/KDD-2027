import os
import shutil
import json

# Paths
BASE_DIR = "./evaluation"
OLD_DATA_DIR = os.path.join(BASE_DIR, "data")
OLD_SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

STRUCTURE = {
    "registry": ["labels.jsonl", "annotations.jsonl", "inventory.json"],
    "cache": ["scrawld", "graphs", "embeddings"],
    "scripts": ["preprocessing", "maintenance"],
    "benchmarks": ["tier1_detection", "tier2_localization", "tier3_perplexity"],
    "results": ["raw", "tables"]
}

FILE_MAP = {
    "unified_labels.jsonl": "registry/labels.jsonl",
    "line_annotations.jsonl": "registry/annotations.jsonl",
    "matching_report.json": "registry/inventory.json",
    "scrawld_sources": "cache/scrawld",
    "scrawld_matched_local.json": "registry/scrawld_matched_local.json",
    "scrawld_missing_sources.json": "registry/scrawld_missing_sources.json"
}

SCRIPT_MAP = {
    "step0a_label_merger.py": "scripts/preprocessing/step0a_label_merger.py",
    "step0b_download_scrawld.py": "scripts/maintenance/step0b_download_scrawld.py",
    "step0b_merge_scrawld.py": "scripts/preprocessing/step0b_merge_scrawld.py",
    "step0c_line_parser.py": "scripts/preprocessing/step0c_line_parser.py",
    "step0d_deduplicator.py": "scripts/preprocessing/step0d_deduplicator.py",
    "step0_scrawld_matcher.py": "scripts/maintenance/step0_scrawld_matcher.py"
}

def main():
    print("=== Evaluation Framework Manager: Restructuring ===")
    
    # 1. Create Directory Tree
    for root, subs in STRUCTURE.items():
        root_path = os.path.join(BASE_DIR, root)
        os.makedirs(root_path, exist_ok=True)
        for sub in subs:
            os.makedirs(os.path.join(root_path, sub), exist_ok=True)
            
    # 2. Migrate Data Files
    for old_name, new_rel in FILE_MAP.items():
        old_path = os.path.join(OLD_DATA_DIR, old_name)
        new_path = os.path.join(BASE_DIR, new_rel)
        
        if os.path.exists(old_path):
            print(f"Moving {old_name} -> {new_rel}")
            if os.path.isdir(old_path):
                # For directories like scrawld_sources
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.move(old_path, new_path)
            else:
                shutil.move(old_path, new_path)

    # 3. Migrate Scripts
    for old_name, new_rel in SCRIPT_MAP.items():
        old_path = os.path.join(OLD_SCRIPTS_DIR, old_name)
        new_path = os.path.join(BASE_DIR, new_rel)
        
        if os.path.exists(old_path):
            print(f"Moving script {old_name} -> {new_rel}")
            shutil.move(old_path, new_path)

    # 4. Handle Symlinks for Graphs
    # Pretraining graphs are in ./Dataset_for_first/datadump_cleaned/features/
    graph_src = "./Dataset_for_first/datadump_cleaned/features"
    graph_dest = os.path.join(BASE_DIR, "cache/graphs/pretrain_shards")
    
    if os.path.exists(graph_src) and not os.path.exists(graph_dest):
        print(f"Creating symlink for graphs: {graph_dest}")
        os.symlink(graph_src, graph_dest)

    print("\n✅ Restructuring Complete.")

if __name__ == "__main__":
    main()
