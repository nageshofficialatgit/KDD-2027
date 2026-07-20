#!/usr/bin/env python3
import os
import json
import glob
from terminaltables import AsciiTable

ROOT_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT_DIR, "Eval_Phase2")
LOGS_DIR = os.path.join(EVAL_DIR, "logs")
REPORTS_DIR = os.path.join(EVAL_DIR, "reports")

def generate_report():
    print("--- Generating Phase 2 Final Report ---")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    table_data = [
        ['Model', 'Top-1 Accuracy', 'Top-3 Accuracy']
    ]
    
    all_results = {}
    
    # First, try to load from JSON reports
    json_files = glob.glob(os.path.join(REPORTS_DIR, "*_localization.json"))
    for file in json_files:
        with open(file, 'r', errors='ignore') as f:
            try:
                data = json.load(f)
                name = data.get("model_name", os.path.basename(file).replace("_localization.json", ""))
                top1 = data.get("top1_acc", 0.0)
                top3 = data.get("top3_acc", 0.0)
                all_results[name] = {"top1": top1, "top3": top3}
            except json.JSONDecodeError:
                pass

    # Then, parse from logs for any missing models or all logs to be safe
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))
    for log_path in log_files:
        name = os.path.basename(log_path).replace(".log", "")
        if name not in all_results:
            top1 = 0.0
            top3 = 0.0
            # Read backwards or just standard read with ignore
            with open(log_path, 'r', errors='ignore') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.startswith("Top-1 Accuracy:"):
                        top1 = float(line.split(":")[1].strip())
                    elif line.startswith("Top-3 Accuracy:"):
                        top3 = float(line.split(":")[1].strip())
                    
                    if top1 > 0 and top3 > 0:
                        break
            
            if top1 > 0 or top3 > 0:
                all_results[name] = {"top1": top1, "top3": top3}

    if not all_results:
        print("No results found in logs or reports.")
        return

    # Prepare table rows and sort by Top-1 Accuracy
    for name, metrics in all_results.items():
        table_data.append([
            name,
            f"{metrics['top1']:.4f}",
            f"{metrics['top3']:.4f}",
        ])
    
    header = table_data[0]
    rows = sorted(table_data[1:], key=lambda x: float(x[1]), reverse=True)
    
    # Output Table
    table = AsciiTable([header] + rows)
    print("\n" + table.table + "\n")
    
    # Save Report
    report_path = os.path.join(REPORTS_DIR, "Table_1_Phase2_Results.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 2 Evaluation Results (MIL Localization)\n\n")
        f.write(f"| {' | '.join(header)} |\n")
        f.write(f"|{'--- |' * len(header)}\n")
        for row in rows:
            f.write(f"| {' | '.join(row)} |\n")
            
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    generate_report()
