import os
import json
import time
import requests
import re
from urllib.parse import urlencode

API_KEY = "EV1R62ZSRBK93TFW74JSBT7XPGTP721YB6"
MISSING_JSON = "./evaluation/data/scrawld_missing_sources.json"
OUTPUT_DIR = "./Dataset_for_first/ScrawlD/contract_sources"

def fetch_source_code(address):
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=contract&action=getsourcecode&address={address}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("status") == "1" and data.get("result"):
            source_content = data["result"][0].get("SourceCode", "")
            if not source_content:
                return None
            
            # If the source is a standard JSON input (starts with {{ or {)
            if source_content.startswith('{{') and source_content.endswith('}}'):
                source_content = source_content[1:-1] # Remove outer braces
                try:
                    json_sources = json.loads(source_content)["sources"]
                    # concat all into one string
                    concat_source = ""
                    for path, content_dict in json_sources.items():
                        concat_source += f"// File: {path}\n"
                        concat_source += content_dict["content"] + "\n\n"
                    return concat_source
                except json.JSONDecodeError:
                    return source_content
            elif source_content.startswith('{') and source_content.endswith('}'):
                try:
                    json_sources = json.loads(source_content)
                    if "sources" in json_sources:
                         json_sources = json_sources["sources"]
                         concat_source = ""
                         for path, content_dict in json_sources.items():
                             concat_source += f"// File: {path}\n"
                             concat_source += content_dict["content"] + "\n\n"
                         return concat_source
                except json.JSONDecodeError:
                    pass
            return source_content
        else:
            print(f"Error fetching {address}: {data.get('message')} - {data.get('result')}")
            return None
    except Exception as e:
        print(f"Exception fetching {address}: {e}")
        return None

def main():
    print(f"=== Downloading {MISSING_JSON} ===")
    
    if not os.path.exists(MISSING_JSON):
        print(f"Missing file not found: {MISSING_JSON}")
        return
        
    with open(MISSING_JSON, 'r') as f:
        missing_files = json.load(f)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Found {len(missing_files)} missing sources to download.")
    count_success = 0
    count_failed = 0
    
    for filename in missing_files:
        # File name like "0x123..._ext.sol"
        address_match = re.search(r'(0x[a-fA-F0-9]{40})', filename)
        if not address_match:
            continue
            
        address = address_match.group(1).lower()
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # Already downloaded
            count_success += 1
            continue
            
        source_code = fetch_source_code(address)
        if source_code:
            with open(filepath, 'w', encoding='utf-8') as sf:
                sf.write(source_code)
            count_success += 1
            print(f"[{count_success}/{len(missing_files)}] Downloaded: {address}")
        else:
            count_failed += 1
            print(f"[{count_success}/{len(missing_files)}] Failed: {address}")
            
        time.sleep(0.25) # Etherscan free tier is 5 req/sec
        
    print(f"--- Download Complete ---")
    print(f"Successfully downloaded: {count_success}")
    print(f"Failed to download: {count_failed}")

if __name__ == "__main__":
    main()
