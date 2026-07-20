import numpy as np
import onnxruntime as ort
import json

class EagleEngine:
    def __init__(self, onnx_path, anchors_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        with open(anchors_path, 'r') as f:
            self.raw_anchors = [json.loads(line) for line in f]
            
        self.anchor_types = [a['type'] for a in self.raw_anchors if not a['is_safe']]
        self.safe_anchor_types = [a['type'] for a in self.raw_anchors if a['is_safe']]
        
        self.vuln_anchors_emb = None
        self.safe_anchors_emb = None
        
    def set_anchors(self, vuln_emb, safe_emb):
        """Sets the L2-normalized anchor embeddings."""
        vuln_np = np.array(vuln_emb, dtype=np.float32)
        safe_np = np.array(safe_emb, dtype=np.float32)
        
        self.vuln_anchors_emb = vuln_np / (np.linalg.norm(vuln_np, axis=-1, keepdims=True) + 1e-9)
        self.safe_anchors_emb = safe_np / (np.linalg.norm(safe_np, axis=-1, keepdims=True) + 1e-9)
        
    def run_chunk(self, chunk):
        inputs = {
            "input_ids": np.array(chunk["input_ids"], dtype=np.int64).reshape(1, -1),
            "attention_mask": np.array(chunk["attention_mask"], dtype=np.int64).reshape(1, -1),
            "token_to_line": np.array(chunk["token_to_line"], dtype=np.int64).reshape(1, -1),
        }
        outs = self.session.run(None, inputs)
        return outs

    def run_document(self, chunks):
        """
        Runs all chunks and merges overlapping results using Center-Weighted Merging (D17).
        """
        line_to_score = {}
        line_to_emb = {}
        line_to_dist = {}
        
        max_bag_prob = 0.0
        
        for chunk in chunks:
            c_len = chunk["end_idx"] - chunk["start_idx"]
            if c_len == 0: continue
            
            center_token_idx = c_len / 2.0
            
            outs = self.run_chunk(chunk)
            line_scores = outs[0][0] # [L]
            bag_logits = outs[1][0]  # [8]
            contrastive_emb = outs[2][0] # [L, 64]
            
            # Sigmoid on the max logit
            chunk_max_prob = 1 / (1 + np.exp(-np.max(bag_logits)))
            if chunk_max_prob > max_bag_prob:
                max_bag_prob = chunk_max_prob
            
            c_lines = chunk["token_to_line"][:c_len]
            unique_lines = set(c_lines)
            if 0 in unique_lines: unique_lines.remove(0)
            
            for line in unique_lines:
                # Find token positions for this line within this chunk
                token_indices = [i for i, l in enumerate(c_lines) if l == line]
                mean_idx = sum(token_indices) / len(token_indices)
                dist_to_center = abs(mean_idx - center_token_idx)
                
                # Update if this chunk provides a more centrally-located context
                if line not in line_to_dist or dist_to_center < line_to_dist[line]:
                    line_to_dist[line] = dist_to_center
                    if line < len(line_scores):
                        line_to_score[line] = line_scores[line]
                        line_to_emb[line] = contrastive_emb[line]
                        
        return line_to_score, line_to_emb, max_bag_prob

    def analyze(self, chunks, source_text):
        """
        Runs the engine and returns diagnostics using Multi-Anchor Cosine Similarity (D14/D18).
        """
        diagnostics = []
        if not chunks: return diagnostics
        
        # Parse source_text for inline suppression
        lines = source_text.splitlines()
        suppressed_lines = set()
        for i, line_str in enumerate(lines):
            if "// eagle-disable-next-line" in line_str:
                suppressed_lines.add(i + 2) # i is 0-indexed, but our line numbers are 1-indexed. Next line is i+2
        
        line_scores, line_embs, max_bag_prob = self.run_document(chunks)
        
        # Flag the Top 10 most suspicious lines (user request)
        top_lines = sorted(line_scores.keys(), key=lambda l: line_scores[l], reverse=True)[:10]
        
        for line in top_lines:
            score = line_scores[line]
            if score > 0.01:  # Absolute noise floor
                emb = line_embs[line]
                # L2 normalize
                emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                
                # Cosine similarity (dot product of unit vectors)
                vuln_sims = np.dot(self.vuln_anchors_emb, emb_norm)
                safe_sims = np.dot(self.safe_anchors_emb, emb_norm)
                
                max_vuln_idx = np.argmax(vuln_sims)
                max_vuln_sim = vuln_sims[max_vuln_idx]
                max_safe_sim = np.max(safe_sims)
                # Flag if closer to vulnerable manifold than safe manifold (D7)
                if max_vuln_sim > max_safe_sim:
                    # UX Severity mapping based on Bag Probability
                    severity = 1 if max_bag_prob > 0.95 else 2 if max_bag_prob > 0.75 else 3
                    
                    if line in suppressed_lines:
                        continue
                        
                    vuln_type = self.anchor_types[max_vuln_idx]
                    diagnostics.append({
                        "line": int(line),
                        "type": vuln_type,
                        "score": float(score),
                        "bag_prob": float(max_bag_prob),
                        "severity": severity,
                        "vuln_sim": float(max_vuln_sim),
                        "safe_sim": float(max_safe_sim)
                    })
                    
        return diagnostics
