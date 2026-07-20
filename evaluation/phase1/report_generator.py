#!/usr/bin/env python3
import os
import json
import glob
from terminaltables import AsciiTable

ROOT_DIR = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT_DIR, "Eval")
METRICS_DIR = os.path.join(EVAL_DIR, "metrics")
REPORTS_DIR = os.path.join(EVAL_DIR, "reports")

def generate_report():
    print("--- Generating Phase 1 Final Report ---")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    table_data = [
        ['Model', 'Macro-F1', 'Avg Precision', 'Avg Recall', 'Avg AUC']
    ]
    
    all_results = {}
    
    # Scrape all summary.json files
    summary_files = glob.glob(os.path.join(METRICS_DIR, "**", "summary.json"), recursive=True)
    
    if not summary_files:
        print("No summary.json files found in Eval/metrics/")
        return

    for fp in summary_files:
        with open(fp, 'r') as f:
            data = json.load(f)
            
        name = data.get("model_name", "Unknown")
        metrics = data.get("test_metrics", {})
        
        macro_f1 = metrics.get("macro_f1", 0.0)
        
        # Calculate averages, safely ignoring NaNs
        precisions = metrics.get("precision", [])
        avg_p = sum(precisions) / len(precisions) if precisions else 0.0
        
        recalls = metrics.get("recall", [])
        avg_r = sum(recalls) / len(recalls) if recalls else 0.0
        
        aucs = [a for a in metrics.get("auc", []) if a is not None and not (isinstance(a, float) and a != a)] # Remove Nones and NaNs
        avg_auc = sum(aucs) / len(aucs) if aucs else 0.0
        
        all_results[name] = data
        
        table_data.append([
            name, 
            f"{macro_f1:.4f}", 
            f"{avg_p:.4f}", 
            f"{avg_r:.4f}", 
            f"{avg_auc:.4f}"
        ])

    # Sort table by Macro-F1 (descending)
    header = table_data[0]
    rows = sorted(table_data[1:], key=lambda x: float(x[1]), reverse=True)
    
    # Output Table
    table = AsciiTable([header] + rows)
    print("\n" + table.table + "\n")
    
    # Save Report
    report_path = os.path.join(REPORTS_DIR, "Table_1_Phase1_Results.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 1 Evaluation Results (Supervised Detection)\n\n")
        
        # Markdown Table Format
        f.write(f"| {' | '.join(header)} |\n")
        f.write(f"|{'|'.join(['---'] * len(header))}|\n")
        for row in rows:
            f.write(f"| {' | '.join(row)} |\n")
            
    print(f"Report saved to: {report_path}")
    
    aggr_json_path = os.path.join(REPORTS_DIR, "aggregate_metrics.json")
    with open(aggr_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw data saved to: {aggr_json_path}")

if __name__ == "__main__":
    generate_report()
