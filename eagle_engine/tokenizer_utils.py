import re

BUCKETS = [256, 512, 1024, 2048, 4096, 8192]
WINDOW_SIZE = 8192
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

def strip_comments_safe(text):
    """
    Safely strips Solidity block and inline comments while preserving string literals.
    Replaces comments with newlines to preserve exact line numbering.
    """
    pattern = r'(".*?"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*)'
    def replacer(match):
        if match.group(1): 
            return match.group(1) # It's a string, keep it
        else:
            # It's a comment, replace with newlines
            return '\n' * match.group(2).count('\n')
    return re.sub(pattern, replacer, text, flags=re.DOTALL)

def build_line_mapping(code_string, offsets):
    """
    Given the full source code and token offsets, returns an array mapping
    each token index to its 1-based line number. 0 is used for special tokens.
    """
    lines = code_string.split('\n')
    line_starts = []
    curr = 0
    for line in lines:
        line_starts.append(curr)
        curr += len(line) + 1 # +1 for \n
        
    def get_line(byte_offset):
        for i in range(len(line_starts)-1, -1, -1):
            if byte_offset >= line_starts[i]:
                return i + 1
        return 1

    token_to_line = []
    for (start, end) in offsets:
        if start == end == 0:
            token_to_line.append(0)
        else:
            token_to_line.append(get_line(start))
    return token_to_line

def prepare_inputs(code_string, tokenizer):
    """
    Strips comments, tokenizes, and maps offsets.
    Returns chunked and padded inputs ready for ONNX.
    """
    clean_code = strip_comments_safe(code_string)
    out = tokenizer(clean_code, return_offsets_mapping=True, truncation=False)
    
    input_ids = out['input_ids']
    attention_mask = out['attention_mask']
    offsets = out['offset_mapping']
    
    token_to_line = build_line_mapping(clean_code, offsets)
    
    T = len(input_ids)
    chunks = []
    
    if T <= WINDOW_SIZE:
        bucket = WINDOW_SIZE
        for b in BUCKETS:
            if T <= b:
                bucket = b
                break
        
        pad_len = bucket - T
        chunk = {
            "input_ids": input_ids + [tokenizer.pad_token_id] * pad_len,
            "attention_mask": attention_mask + [0] * pad_len,
            "token_to_line": token_to_line + [0] * pad_len,
            "start_idx": 0,
            "end_idx": T
        }
        chunks.append(chunk)
    else:
        # Center-weighted Sliding window
        start = 0
        while start < T:
            end = min(start + WINDOW_SIZE, T)
            c_ids = input_ids[start:end]
            c_mask = attention_mask[start:end]
            c_line = token_to_line[start:end]
            
            c_len = len(c_ids)
            # For chunks that are exactly WINDOW_SIZE, padding is 0.
            # For the last chunk, it might be less. Find bucket.
            bucket = WINDOW_SIZE
            pad_len = bucket - c_len
            
            chunk = {
                "input_ids": c_ids + [tokenizer.pad_token_id] * pad_len,
                "attention_mask": c_mask + [0] * pad_len,
                "token_to_line": c_line + [0] * pad_len,
                "start_idx": start,
                "end_idx": end
            }
            chunks.append(chunk)
            if end == T: break
            start += STRIDE
            
    return chunks, token_to_line, T
