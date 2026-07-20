#!/usr/bin/env python3
import re
import random
import json
import os

class SolidityPerturber:
    """
    Applies semantic-preserving transformations to Solidity code to test model robustness.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        
    def rename_identifiers(self, code):
        """
        Renames variables, functions, and events to generic names (var1, func1, etc.)
        """
        # Find all identifiers (basic regex, avoids keywords)
        keywords = {
            "pragma", "solidity", "contract", "library", "interface", "is", "using", "for",
            "function", "modifier", "event", "struct", "enum", "mapping", "address", "uint",
            "uint256", "int", "bool", "string", "bytes", "byte", "public", "private", "external",
            "internal", "pure", "view", "payable", "returns", "return", "if", "else", "while",
            "for", "do", "break", "continue", "throw", "revert", "require", "assert", "emit",
            "storage", "memory", "calldata", "msg", "tx", "block", "now", "super", "selfdestruct",
            "constant", "indexed", "new", "true", "false"
        }
        
        # Regex for identifiers: starts with a-zA-Z_ followed by alnum
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = set(re.findall(pattern, code))
        
        # Filter out keywords and numeric-only strings
        targets = [i for i in identifiers if i not in keywords and not i.isdigit() and len(i) > 1]
        
        # Create mapping
        mapping = {}
        for idx, target in enumerate(sorted(targets)):
            mapping[target] = f"id_{idx:03d}"
            
        # Replace (longest first to avoid partial matches)
        sorted_targets = sorted(targets, key=len, reverse=True)
        perturbed_code = code
        for target in sorted_targets:
            # Use word boundary to ensure we don't match inside other words
            perturbed_code = re.sub(rf'\b{target}\b', mapping[target], perturbed_code)
            
        return perturbed_code, mapping

    def flatten_control_flow(self, code):
        """
        Basic flattening: Replaces 'if(x) { a; } else { b; }' with 'x ? a : b' where possible
        or reorders blocks. (Simplified for this PoC)
        """
        # Placeholder for more complex AST-based flattening
        # For now, let's just add 'opaque' comments that break syntax patterns
        lines = code.splitlines()
        new_lines = []
        for line in lines:
            if self.rng.random() > 0.7:
                new_lines.append(line + " // op_nop")
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def perturb(self, code):
        code, mapping = self.rename_identifiers(code)
        code = self.flatten_control_flow(code)
        return code

def generate_adversarial_subset(input_labels, input_annos, output_labels, output_annos, limit=100):
    perturber = SolidityPerturber()
    
    # Process Labels
    count = 0
    perturbed_filenames = {} # orig -> perturbed
    
    with open(input_labels, 'r') as f_in, open(output_labels, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            orig_fn = data['filename']
            perturbed_fn = orig_fn + ".perturbed"
            perturbed_filenames[orig_fn] = perturbed_fn
            
            perturbed_src = perturber.perturb(data['source_code'])
            data['source_code'] = perturbed_src
            data['filename'] = perturbed_fn
            
            f_out.write(json.dumps(data) + "\n")
            count += 1
            if count >= limit: break
            
    # Process Annotations
    with open(input_annos, 'r') as f_in, open(output_annos, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            if data['filename'] in perturbed_filenames:
                data['filename'] = perturbed_filenames[data['filename']]
                f_out.write(json.dumps(data) + "\n")
                
    print(f"Generated {count} perturbed samples and matching annotations.")

if __name__ == "__main__":
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    REGISTRY_DIR = os.path.join(PROJECT_ROOT, "Implementation/evaluation/registry")
    
    # Process ALL labels and annotations
    label_files = [os.path.join(REGISTRY_DIR, f) for f in os.listdir(REGISTRY_DIR) if "labels" in f]
    anno_files = [os.path.join(REGISTRY_DIR, f) for f in os.listdir(REGISTRY_DIR) if "annotations" in f]
    
    OUT_LABELS = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/adversarial_labels.jsonl")
    OUT_ANNOS = os.path.join(PROJECT_ROOT, "Implementation/evaluation/semantic/adversarial_annotations.jsonl")
    
    os.makedirs(os.path.dirname(OUT_LABELS), exist_ok=True)
    
    # Combined processing
    perturber = SolidityPerturber()
    perturbed_filenames = {}
    count = 0
    
    with open(OUT_LABELS, 'w') as f_out_l:
        for lf in label_files:
            with open(lf, 'r') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    orig_fn = data['filename']
                    perturbed_fn = orig_fn + ".perturbed"
                    perturbed_filenames[orig_fn] = perturbed_fn
                    data['source_code'] = perturber.perturb(data['source_code'])
                    data['filename'] = perturbed_fn
                    f_out_l.write(json.dumps(data) + "\n")
                    count += 1
                    if count >= 200: break # Increased limit
            if count >= 200: break
            
    with open(OUT_ANNOS, 'w') as f_out_a:
        for af in anno_files:
            if not os.path.exists(af): continue
            with open(af, 'r') as f_in:
                for line in f_in:
                    data = json.loads(line)
                    if data['filename'] in perturbed_filenames:
                        data['filename'] = perturbed_filenames[data['filename']]
                        f_out_a.write(json.dumps(data) + "\n")
                        
    print(f"Generated {count} perturbed samples and matching annotations.")
