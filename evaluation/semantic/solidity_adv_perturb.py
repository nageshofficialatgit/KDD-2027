#!/usr/bin/env python3
import re
import random
import json
import os

class AdvancedSolidityPerturber:
    """
    Applies robust syntactic transformations that preserve block structure.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def perturb(self, source):
        lines = source.splitlines()
        new_lines = []
        mapping = {} 
        
        for i, line in enumerate(lines):
            new_line = line
            
            # 1. require(cond); -> if (!(cond)) revert();
            # Only do this if it's a simple line to avoid complex regex failures
            if "require(" in line and ";" in line and self.rng.random() > 0.5:
                # Find the innermost parentheses
                m = re.search(r'require\((.*)\);', line)
                if m:
                    cond_full = m.group(1).strip()
                    # Strip message if present (last comma)
                    # Note: this is a heuristic, but safer
                    if "," in cond_full and "\"" in cond_full:
                         # Assume the last part is the message
                         cond = cond_full.rsplit(",", 1)[0].strip()
                    else:
                         cond = cond_full
                    
                    new_line = line.replace(f"require({cond_full});", f"if (!({cond})) revert();")

            mapping[i + 1] = [len(new_lines) + 1]
            new_lines.append(new_line)

        return "\n".join(new_lines), mapping

def process_registry():
    perturber = AdvancedSolidityPerturber()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    REGISTRY_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry")
    OUT_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/ambiguity")
    os.makedirs(OUT_DIR, exist_ok=True)

    label_files = [f for f in os.listdir(REGISTRY_DIR) if "labels.jsonl" in f]
    
    for lf in label_files:
        af = lf.replace("labels", "annotations")
        af_path = os.path.join(REGISTRY_DIR, af)
        lf_path = os.path.join(REGISTRY_DIR, lf)
        if not os.path.exists(af_path): continue
        
        print(f"Processing {lf}...")
        annos_by_file = {}
        with open(af_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                annos_by_file[d['filename']] = d

        out_lf = os.path.join(OUT_DIR, lf.replace(".jsonl", "_ambiguous.jsonl"))
        out_af = os.path.join(OUT_DIR, af.replace(".jsonl", "_ambiguous.jsonl"))

        with open(lf_path, 'r') as f_in, open(out_lf, 'w') as f_out_l, open(out_af, 'w') as f_out_a:
            for line in f_in:
                data = json.loads(line)
                fn = data['filename']
                new_src, l_map = perturber.perturb(data['source_code'])
                new_fn = fn + ".ambiguous"
                data['filename'] = new_fn
                data['source_code'] = new_src
                f_out_l.write(json.dumps(data) + "\n")
                if fn in annos_by_file:
                    anno = annos_by_file[fn]
                    new_vuln = []
                    for v in anno['vulnerable_lines']:
                        if v in l_map: new_vuln.extend(l_map[v])
                    anno_new = anno.copy()
                    anno_new['filename'] = new_fn
                    anno_new['vulnerable_lines'] = sorted(list(set(new_vuln)))
                    f_out_a.write(json.dumps(anno_new) + "\n")

if __name__ == "__main__":
    process_registry()
