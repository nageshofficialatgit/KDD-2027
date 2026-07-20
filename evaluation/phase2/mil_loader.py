import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import glob

# Ensure Implementation/ is in path for model imports
sys.path.append("./Implementation")

class MILLoader(Dataset):
    """
    Dataset for Multi-Instance Learning localization.
    Supports token-to-line mapping for interpreting attention weights.
    """
    def __init__(self, label_files, annotation_files, tokenizer, max_len=2048, npz_dir=None, df_dir=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.annotations = {} # filename -> list of vulnerable lines
        
        # Load Labels
        for lf in label_files:
            print(f"Loading labels from {lf}...")
            with open(lf, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        
        # Load Annotations (Line-level ground truth)
        for af in annotation_files:
            if os.path.exists(af):
                print(f"Loading annotations from {af}...")
                with open(af, 'r') as f:
                    for line in f:
                        d = json.loads(line)
                        self.annotations[d['filename']] = d['vulnerable_lines']
        
        self.vuln_types = [
            "Reentrancy", "Integer_Overflow_Underflow", "Access_Control", 
            "Timestamp_Dependence", "Transaction_Ordering_Dependence_Front_Running", 
            "tx_origin", "Bad_Randomness", "Unchecked_Low_Level_Calls"
        ]
        
        # Graph feature indexing (optional/targeted)
        self.npz_index = {}
        if npz_dir:
            print(f"Indexing graph shards in {npz_dir}...")
            shard_files = sorted(glob.glob(os.path.join(npz_dir, "*_shard_*.npz")))
            for sf in shard_files:
                try:
                    d = np.load(sf, allow_pickle=True)
                    manifest = json.loads(d["manifest"][0])
                    for entry in manifest:
                        self.npz_index[entry["contract_id"]] = (sf, entry["idx"])
                    d.close()
                except:
                    continue
            print(f"Indexed {len(self.npz_index)} graph-ready contracts.")

        # Data flow indexing (GraphCodeBERT)
        self.df_dir = df_dir
        self.df_index = set()
        if df_dir:
            print(f"Checking data flow cache in {df_dir}...")
            self.df_index = {f.replace(".df.json", "") for f in os.listdir(df_dir) if f.endswith(".df.json")}
            print(f"Found {len(self.df_index)} data flow entries.")

    def __len__(self):
        return len(self.data)

    def _get_graph_features(self, contract_id, encoding):
        graph_feats = np.zeros((self.max_len, 6), dtype=np.float32)
        if contract_id not in self.npz_index:
            return graph_feats
            
        shard_path, ci = self.npz_index[contract_id]
        shard = np.load(shard_path, allow_pickle=True)
        
        node_features = shard[f"node_features_{ci}"]
        token_to_node = shard[f"token_to_node_{ci}"]
        
        if not np.isfinite(node_features).all():
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            
        word_ids = encoding.word_ids(0)
        T_raw = len(token_to_node)
        
        for i in range(min(self.max_len, len(word_ids))):
            word_idx = word_ids[i]
            if word_idx is not None and word_idx < T_raw:
                nid = token_to_node[word_idx]
                if nid >= 0:
                    graph_feats[i] = node_features[nid]
        
        shard.close()
        return graph_feats

    def _tokenize_graphcodebert(self, source, filename):
        # 1. Tokenize source
        code_tokens = self.tokenizer.tokenize(source)[:self.max_len - 4] # Leave room for [SEP] and variables
        
        # 2. Load data flow
        variables = []
        dfg_edges = []
        if filename in self.df_index:
            try:
                with open(os.path.join(self.df_dir, f"{filename}.df.json"), "r") as f:
                    df_data = json.load(f)
                    variables = df_data["variable_sequence"]
                    dfg_edges = df_data["dfg_edges"]
            except:
                pass
        
        # 3. Prepare sequence: [CLS] code [SEP] vars [SEP]
        # Limit variables to fit in max_len
        max_vars = self.max_len - len(code_tokens) - 3
        variables = variables[:max_vars]
        
        var_tokens = [v["name"] for v in variables]
        
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token] + var_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Pad
        padding_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        attention_mask = [1] * len(tokens) + [0] * padding_len
        
        # DFG edges mapping
        # In GraphCodeBERT, DFG edges are between variable tokens.
        # Var tokens start at index len(code_tokens) + 2
        var_start = len(code_tokens) + 2
        dfg_matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        for u, v in dfg_edges:
            if u < max_vars and v < max_vars:
                dfg_matrix[var_start + u, var_start + v] = 1
                dfg_matrix[var_start + v, var_start + u] = 1 # undirected for simplicity in some variants
                
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(dfg_matrix)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item['source_code']
        labels = [item['labels'][v] for v in self.vuln_types]
        filename = item['filename']
        
        if self.df_dir:
            input_ids, attention_mask, dfg_matrix = self._tokenize_graphcodebert(src, filename)
            # For GraphCodeBERT, we still need offsets for the code part to map points back to lines
            # We'll do a partial encoding to get the offsets for code_tokens
            # (Note: this is slightly inefficient but ensures accuracy)
            code_encoding = self.tokenizer(
                src,
                max_length=self.max_len - 3, # Approximate
                truncation=True,
                return_offsets_mapping=True
            )
            code_offsets = code_encoding["offset_mapping"]
            
            # Map offsets to the full sequence
            full_offsets = [(0, 0)] # CLS
            full_offsets.extend(code_offsets[1:-1]) # Tokens
            full_offsets.append((0, 0)) # SEP
            for v in range(len(input_ids) - len(full_offsets)):
                full_offsets.append((0, 0)) # Variables and Padding
            offsets = np.array(full_offsets[:self.max_len])
        else:
            encoding = self.tokenizer(
                src,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            offsets = encoding["offset_mapping"].squeeze(0).numpy()
            dfg_matrix = torch.zeros((self.max_len, self.max_len))
        
        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "filename": filename,
            "dfg": dfg_matrix
        }
        
        # Include token-to-line mapping for evaluation
        token_to_line = []
        line_boundaries = [0]
        for line in src.splitlines(keepends=True):
            line_boundaries.append(line_boundaries[-1] + len(line))
            
        for start, end in offsets:
            if start == 0 and end == 0:
                token_to_line.append(-1)
                continue
                
            line_idx = -1
            for i in range(len(line_boundaries) - 1):
                if start >= line_boundaries[i] and start < line_boundaries[i+1]:
                    line_idx = i + 1
                    break
            token_to_line.append(line_idx)
            
        res["token_to_line"] = torch.tensor(token_to_line, dtype=torch.long)
        
        # Create line sink mask: only the FIRST token of each valid line gets a 1
        line_sink_mask = torch.zeros(self.max_len, dtype=torch.long)
        seen_lines = set()
        for i, l_idx in enumerate(token_to_line):
            if l_idx > 0 and l_idx not in seen_lines:
                seen_lines.add(l_idx)
                line_sink_mask[i] = 1
        
        res["line_sink_mask"] = line_sink_mask
        
        if self.npz_index:
             # We need encoding for offsets in _get_graph_features
             # If we used self.df_dir, we didn't have a full encoding object.
             # Let's fix _get_graph_features to take offsets directly or re-run tokenizer
             # Since WIRE doesn't use GraphCodeBERT features (usually), this is fine.
             temp_enc = self.tokenizer(src, max_length=self.max_len, truncation=True, padding="max_length")
             g_feats = self._get_graph_features(filename, temp_enc)
             res["graph_features"] = torch.from_numpy(g_feats).float()
             
        return res

def get_mil_splits(label_files, test_subset_dataset="ScrawlD", seed=42):
    """
    Splits data for MIL. 
    Following user request: Train on all, Test on ScrawlD subset.
    """
    import random
    rng = random.Random(seed)
    
    all_data = []
    for lf in label_files:
        with open(lf, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
    
    # Identify ScrawlD entries for the test set
    scrawld_indices = [i for i, d in enumerate(all_data) if d.get("dataset") == "ScrawlD"]
    other_indices = [i for i, d in enumerate(all_data) if d.get("dataset") != "ScrawlD"]
    
    rng.shuffle(scrawld_indices)
    
    # We want a clean test set from ScrawlD
    n_test = len(scrawld_indices) // 5 # 20% of ScrawlD for test
    test_indices = scrawld_indices[:n_test]
    val_indices = scrawld_indices[n_test : n_test*2]
    
    # Training includes ALL of 'other' and the rest of 'ScrawlD'
    train_indices = other_indices + scrawld_indices[n_test*2:]
    
    return train_indices, val_indices, test_indices
