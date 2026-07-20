#!/usr/bin/env python3
"""
Graph-RoPE Sentry: Continuous Training & Data Health Monitor.

Runs on CPU in the background to detect:
1. Training Instability (Loss spikes, gradient explosion) - checked every minute
2. Model Collapse (Attention entropy drop, feature underutilization) - checked every minute
3. Dataset Issues (Feature variance, label noise, edge leakage) - checked every 6 hours

Usage:
    python sentry_graph_rope.py --config config_e1.json
"""

import argparse
import csv
import json
import math
import os
import time
import sys
import numpy as np
import torch
from pathlib import Path
from collections import deque

# ANSI colors for console
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

class Sentry:
    def __init__(self, log_dir, feat_dir, log_interval=60, data_interval=21600):
        self.log_dir = Path(log_dir)
        self.feat_dir = Path(feat_dir)
        self.log_interval = log_interval
        self.data_interval = data_interval
        
        self.csv_path = self.log_dir / "metrics.csv"
        # Alert log is jsonl
        self.alert_path = self.log_dir / "sentry_alerts.jsonl"
        self.last_step_checked = -1
        
        # History for trend analysis
        self.loss_history = deque(maxlen=20)
        self.grad_history = deque(maxlen=20)

    def log_alert(self, msg, level="INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": msg
        }
        
        # Console: Only Warnings/Critical
        if level in ["WARNING", "CRITICAL"] or "Sentry active" in msg:
            color = GREEN
            if level == "WARNING": color = YELLOW
            if level == "CRITICAL": color = RED
            print(f"{color}[{timestamp}] {level}: {msg}{RESET}", flush=True)
            
        # File: Everything (JSONL)
        try:
            with open(self.alert_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Failed to write to alert log: {e}")

    def check_training_metrics(self):
        """Analyze metrics.csv for instability and collapse."""
        if not self.csv_path.exists():
            return

        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return

                # Only check new rows
                new_rows = [r for r in rows if int(r["step"]) > self.last_step_checked]
                if not new_rows:
                    return

                def safe_float(v, default=0.0):
                    try:
                        return float(v) if v and v != "NaN" else default
                    except (ValueError, TypeError):
                        return default

                for row in new_rows:
                    step = int(row["step"])
                    self.last_step_checked = step
                    
                    # 1. Loss Instability
                    loss = safe_float(row.get("train/loss"))
                    if loss > 0: # Only check if loss is reported in this row
                        if math.isnan(loss) or math.isinf(loss):
                            self.log_alert(f"Step {step}: Loss is NaN/Inf!", "CRITICAL")
                        self.loss_history.append(loss)
                        
                        if len(self.loss_history) > 5:
                            avg_loss = np.mean(list(self.loss_history)[:-1])
                            if avg_loss > 0.1 and loss > 3 * avg_loss:
                                self.log_alert(f"Step {step}: Sudden loss spike ({loss:.4f})", "WARNING")

                    # 2. Gradient Explosion
                    grad = safe_float(row.get("train/grad_norm"))
                    if grad > 0:
                        self.grad_history.append(grad)
                        if grad > 10.0:
                            self.log_alert(f"Step {step}: Gradient explosion (norm={grad:.2f})", "CRITICAL")
                    
                    # 3. Attention Entropy
                    for k in row:
                        if k.startswith("entropy/") and row[k]:
                            val = float(row[k])
                            if val < 0.1:
                                self.log_alert(f"Step {step}: Attention collapse in {k} ({val:.4f})", "CRITICAL")

                    # 4. Feature Utilization
                    for k in row:
                        if k.startswith("rot/util_") and row[k]:
                            val = float(row[k])
                            if val < 0.05:
                                self.log_alert(f"Step {step}: Feature collapse in {k} ({val:.3f})", "WARNING")

        except Exception as e:
            self.log_alert(f"Error reading metrics: {e}", "WARNING")

    def check_dataset_health(self):
        """Randomly sample NPZ shards to check feature quality."""
        shards = list(self.feat_dir.glob("*.npz"))
        if not shards:
            self.log_alert("No feature shards found!", "WARNING")
            return

        # Sample 1 random shard
        shard = np.random.choice(shards)
        try:
            self.log_alert(f"Deep Scan: Checking {shard.name}...", "INFO")
            data = np.load(shard, allow_pickle=True)
            
            keys = list(data.keys())
            if not keys: return
            
            sample_keys = np.random.choice(keys, min(5, len(keys)), replace=False)
            
            degenerate_count = 0
            for cid in sample_keys:
                feats = data[cid]
                
                # Check for object arrays (lists of inconsistent length or strings)
                if feats.dtype == object:
                    try:
                        feats = feats.astype(float)
                    except Exception:
                        self.log_alert(f"Non-numeric features in {cid} (dtype=object)", "WARNING")
                        continue

                if not np.isfinite(feats).all():
                    self.log_alert(f"NaN/Inf features in {cid}", "CRITICAL")
                
                std = feats.std(axis=0)
                if (std < 0.001).any():
                    degenerate_count += 1
                    bad_idx = np.where(std < 0.001)[0]
                    self.log_alert(f"Contract {cid}: Zero variance in features {bad_idx}", "WARNING")
            
            if degenerate_count > 2:
                self.log_alert(f"Shard {shard.name} has >40% degenerate contracts", "WARNING")
            else:
                self.log_alert(f"Deep Scan: {shard.name} PASSED", "INFO")
                
        except Exception as e:
            self.log_alert(f"Error checking dataset: {e}", "WARNING")

    def generate_dashboard(self):
        """Generate a self-contained, reactive Plotly dashboard.
        
        Auto-detects available metrics from CSV columns and conditionally
        renders charts only for metrics that have data. Works for both
        baseline (Graph-RoPE) and WIRE ablation experiments.
        """
        if not self.csv_path.exists():
            return

        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return
        except Exception as e:
            self.log_alert(f"Dashboard error reading CSV: {e}", "WARNING")
            return

        # Prepare metrics data for JS
        steps = []
        metrics_json = {}
        for row in rows:
            try:
                s = int(row["step"])
                steps.append(s)
                for k, v in row.items():
                    if k == "step": continue
                    if k not in metrics_json: metrics_json[k] = []
                    try:
                        metrics_json[k].append(float(v) if v and v != "NaN" else None)
                    except ValueError:
                        metrics_json[k].append(None)
            except (KeyError, ValueError):
                continue
        
        if not steps: return

        # Auto-detect which metrics have actual data
        available = set()
        for k, vals in metrics_json.items():
            if any(v is not None for v in vals):
                available.add(k)

        has_node_loss = "train/node_loss" in available
        has_err_loss = "train/err_loss" in available
        has_perplexity = "train/perplexity" in available
        has_lr = "train/lr_encoder" in available
        has_wall_time = "train/wall_time_min" in available
        has_util = any(k.startswith("rot/util_") for k in available)
        has_entropy = any(k.startswith("xai/entropy_") for k in available)
        has_val = "val/loss" in available

        # Load alerts
        alerts = []
        if self.alert_path.exists():
            try:
                with open(self.alert_path) as f:
                    for line in f.readlines()[-50:]:
                        try: alerts.append(json.loads(line.strip()))
                        except: continue
            except: pass

        # Load config for total_steps / epochs if available
        config_path = self.log_dir / "config.json"
        total_steps = steps[-1]  # fallback
        total_epochs = "?"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                total_steps = cfg.get("_computed", {}).get("total_steps", cfg.get("total_steps", steps[-1]))
                total_epochs = cfg.get("training", {}).get("epochs", "?")
        except: pass

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = os.path.basename(self.log_dir)

        # ── Determine dynamic chart panels ──────────────────────────
        # Each panel: (div_id, css_class)
        chart_panels = [
            ("chart-loss", "full-width"),   # Always
            ("chart-acc", ""),              # Always
            ("chart-grad", ""),             # Always
        ]
        if has_perplexity:
            chart_panels.append(("chart-ppl", ""))
        if has_lr:
            chart_panels.append(("chart-lr", ""))
        if has_util:
            chart_panels.append(("chart-util", ""))
        if has_entropy:
            chart_panels.append(("chart-entropy", "full-width"))

        # Always add alerts panel
        chart_panels.append(("alert-panel", ""))

        # Build panel HTML
        panel_html_parts = []
        for div_id, css_cls in chart_panels:
            cls = f'panel {css_cls}'.strip()
            if div_id == "alert-panel":
                panel_html_parts.append(
                    f'<div class="{cls}">'
                    '<div class="panel-title">Recent Alerts</div>'
                    '<div class="alert-scroll" id="alert-list"></div>'
                    '</div>'
                )
            else:
                panel_html_parts.append(f'<div class="{cls}" id="{div_id}"></div>')
        panels_html = "\n        ".join(panel_html_parts)

        # ── Compute KPI values ──────────────────────────────────────
        def get_last_nonnull(key):
            vals = metrics_json.get(key, [])
            for v in reversed(vals):
                if v is not None: return v
            return None

        current_step = steps[-1]
        progress_pct = min(100, current_step / max(total_steps, 1) * 100) if isinstance(total_steps, (int, float)) else 0
        wall_min = get_last_nonnull("train/wall_time_min")
        wall_str = ""
        eta_str = "—"
        if wall_min is not None:
            hours = wall_min / 60
            wall_str = f"{hours:.1f}h"
            if isinstance(total_steps, (int, float)) and current_step > 0:
                remaining_steps = total_steps - current_step
                eta_min = wall_min / current_step * remaining_steps
                eta_str = f"{eta_min/60:.1f}h" if eta_min > 60 else f"{eta_min:.0f}m"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph-RoPE Sentry — {experiment_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-base:    #111827;
            --bg-surface: #1f2937;
            --bg-elevated:#374151;
            --border:     #374151;
            --border-subtle: #2d3748;
            --text-primary:   #f3f4f6;
            --text-secondary: #9ca3af;
            --text-muted:     #6b7280;
            --emerald:  #34d399;
            --amber:    #fbbf24;
            --sky:      #38bdf8;
            --violet:   #a78bfa;
            --rose:     #fb7185;
            --slate:    #94a3b8;
            --radius:   12px;
            --radius-sm: 8px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            padding: 24px;
            min-height: 100vh;
        }}

        /* ── Header ─────────────────────────────────────── */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, var(--bg-surface) 0%, #1a2332 100%);
            padding: 24px 28px;
            border-radius: var(--radius);
            margin-bottom: 24px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--emerald), var(--sky), var(--violet));
        }}
        .header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
        .header h1 span {{ color: var(--sky); }}
        .header-meta {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .header-right {{ text-align: right; }}
        .step-badge {{
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .status-pill {{
            display: inline-block;
            margin-top: 6px;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: rgba(52, 211, 153, 0.15);
            color: var(--emerald);
            border: 1px solid rgba(52, 211, 153, 0.3);
        }}

        /* ── Progress Bar ───────────────────────────────── */
        .progress-container {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 16px 20px;
            margin-bottom: 24px;
        }}
        .progress-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 12px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .progress-bar {{
            height: 8px;
            background: var(--bg-elevated);
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--sky), var(--violet));
            transition: width 0.5s ease;
        }}

        /* ── Stats Bar ──────────────────────────────────── */
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 4px; height: 100%;
        }}
        .stat-card:nth-child(1)::before {{ background: var(--rose); }}
        .stat-card:nth-child(2)::before {{ background: var(--emerald); }}
        .stat-card:nth-child(3)::before {{ background: var(--violet); }}
        .stat-card:nth-child(4)::before {{ background: var(--sky); }}
        .stat-card:nth-child(5)::before {{ background: var(--amber); }}
        .stat-card:nth-child(6)::before {{ background: var(--slate); }}
        .stat-card:nth-child(7)::before {{ background: var(--emerald); }}
        .stat-card:nth-child(8)::before {{ background: var(--rose); }}
        .stat-label {{
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 22px; font-weight: 700;
            font-family: 'JetBrains Mono', monospace; color: var(--text-primary);
        }}
        .stat-sub {{
            font-size: 11px; color: var(--text-muted); margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* ── Chart Grid ─────────────────────────────────── */
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .full-width {{ grid-column: span 2; }}
        .panel {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            min-height: 380px;
        }}
        .panel-title {{
            font-size: 13px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; color: var(--text-secondary);
            margin-bottom: 12px; padding-bottom: 10px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        /* ── Alerts ──────────────────────────────────────── */
        .alert-scroll {{
            max-height: 340px; overflow-y: auto;
            scrollbar-width: thin; scrollbar-color: var(--bg-elevated) transparent;
        }}
        .alert-item {{
            display: flex; align-items: flex-start; gap: 10px;
            font-size: 12px; padding: 10px 12px; border-radius: 6px;
            margin-bottom: 6px; font-family: 'JetBrains Mono', monospace;
            line-height: 1.5; background: rgba(255,255,255,0.02);
            border-left: 3px solid transparent;
        }}
        .alert-item.level-CRITICAL {{
            color: var(--rose); border-left-color: var(--rose);
            background: rgba(251, 113, 133, 0.06);
        }}
        .alert-item.level-WARNING {{
            color: var(--amber); border-left-color: var(--amber);
            background: rgba(251, 191, 36, 0.06);
        }}
        .alert-item.level-INFO {{
            color: var(--text-muted); border-left-color: var(--border);
        }}
        .alert-ts {{ color: var(--text-muted); white-space: nowrap; flex-shrink: 0; }}

        .footer {{
            text-align: center; color: var(--text-muted); font-size: 11px;
            margin-top: 32px; padding-top: 16px;
            border-top: 1px solid var(--border-subtle);
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .full-width {{ grid-column: span 1; }}
            .stats-bar {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>\U0001f6e1\ufe0f Graph-RoPE <span>Sentry</span></h1>
            <div class="header-meta">{experiment_name} &nbsp;\xb7&nbsp; {timestamp}</div>
        </div>
        <div class="header-right">
            <div class="step-badge">{current_step} / {total_steps}</div>
            <div class="status-pill">\u25cf Training Active</div>
        </div>
    </div>

    <div class="progress-container">
        <div class="progress-header">
            <span>Epoch {get_last_nonnull('train/epoch') or '?'} / {total_epochs}</span>
            <span>{progress_pct:.1f}% &nbsp;\xb7&nbsp; {wall_str or '—'} elapsed &nbsp;\xb7&nbsp; ETA {eta_str}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_pct:.1f}%"></div>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-label">Best Val Loss</div>
            <div class="stat-value" id="stat-val-loss">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Train Loss</div>
            <div class="stat-value" id="stat-loss">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">MLM Accuracy</div>
            <div class="stat-value" id="stat-mlm">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">{"Perplexity" if has_perplexity else "Val MLM Acc"}</div>
            <div class="stat-value" id="stat-ppl">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">ERR Loss</div>
            <div class="stat-value" id="stat-err">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Grad Norm</div>
            <div class="stat-value" id="stat-grad">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">GPU Memory</div>
            <div class="stat-value" id="stat-gpu">—</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">{"Training Time" if has_wall_time else "Val MLM Acc"}</div>
            <div class="stat-value" id="stat-time">—</div>
        </div>
    </div>

    <div class="grid">
        {panels_html}
    </div>

    <div class="footer">
        Auto-refresh 30s &nbsp;\xb7&nbsp; Plotly.js interactive charts
        &nbsp;\xb7&nbsp; Graph-RoPE BTP &copy; 2026
    </div>

    <script>
        const steps = {json.dumps(steps)};
        const metrics = {json.dumps(metrics_json)};
        const alerts = {json.dumps(alerts)};

        function getLast(key) {{
            const arr = metrics[key];
            if (!arr) return null;
            for (let i = arr.length - 1; i >= 0; i--) {{
                if (arr[i] !== null && arr[i] !== undefined) return arr[i];
            }}
            return null;
        }}

        function hasData(key) {{
            const arr = metrics[key];
            return arr && arr.some(v => v !== null && v !== undefined);
        }}

        function createTrace(key, name, color, opts) {{
            if (!hasData(key)) return null;
            return Object.assign({{
                x: steps, y: metrics[key], name: name,
                type: 'scatter', mode: 'lines',
                line: {{ color: color, width: 2.5, shape: 'spline' }},
                connectgaps: true,
            }}, opts || {{}});
        }}

        const C = {{
            rose: '#fb7185', emerald: '#34d399', sky: '#38bdf8',
            violet: '#a78bfa', amber: '#fbbf24', slate: '#94a3b8',
            indigo: '#818cf8', teal: '#2dd4bf',
        }};

        const baseLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor:  'rgba(0,0,0,0)',
            font: {{ family: 'Inter, sans-serif', color: '#9ca3af', size: 12 }},
            margin: {{ t: 48, b: 44, l: 56, r: 24 }},
            xaxis: {{ gridcolor: '#1f2937', zerolinecolor: '#374151', linecolor: '#374151' }},
            yaxis: {{ gridcolor: '#1f2937', zerolinecolor: '#374151', linecolor: '#374151' }},
            legend: {{ orientation: 'h', y: 1.12, font: {{ size: 11 }}, bgcolor: 'rgba(0,0,0,0)' }},
            hoverlabel: {{ bgcolor: '#1f2937', font: {{ family: 'JetBrains Mono', size: 12 }} }},
        }};
        const plotCfg = {{ responsive: true, displayModeBar: false }};

        // ── 1. Loss Convergence (auto-detect loss traces) ────────
        const lossTraces = [
            createTrace('train/loss',      'Total Loss',  C.rose),
            createTrace('train/mlm_loss',  'MLM Loss',    C.sky),
            createTrace('train/node_loss', 'TFR (Node)',   C.violet),
            createTrace('train/err_loss',  'ERR Loss',     C.emerald),
            createTrace('val/loss',        'Val Loss',     '#ffffff', {{ line: {{ dash: 'dash', width: 2 }} }}),
            createTrace('val/mlm_loss',    'Val MLM',      C.slate, {{ line: {{ dash: 'dot', width: 1.5 }} }}),
        ].filter(t => t);
        if (lossTraces.length > 0) {{
            Plotly.newPlot('chart-loss', lossTraces, Object.assign({{}}, baseLayout, {{
                title: {{ text: 'Loss Convergence', font: {{ size: 14, color: '#f3f4f6' }} }},
                yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Loss' }}),
                xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Step' }}),
            }}), plotCfg);
        }}

        // ── 2. Accuracy ─────────────────────────────────────────
        const accTraces = [
            createTrace('train/mlm_acc', 'MLM Train', C.sky),
            createTrace('val/mlm_acc',   'MLM Val',   '#ffffff', {{ line: {{ dash: 'dash', width: 2 }} }}),
        ].filter(t => t);
        if (accTraces.length > 0) {{
            Plotly.newPlot('chart-acc', accTraces, Object.assign({{}}, baseLayout, {{
                title: {{ text: 'MLM Accuracy', font: {{ size: 14, color: '#f3f4f6' }} }},
                yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Accuracy', range: [0, 1] }}),
                xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Step' }}),
            }}), plotCfg);
        }}

        // ── 3. Gradient Stability ───────────────────────────────
        const gradTraces = [createTrace('train/grad_norm', 'Grad Norm', C.amber)].filter(t => t);
        if (gradTraces.length > 0) {{
            Plotly.newPlot('chart-grad', gradTraces, Object.assign({{}}, baseLayout, {{
                title: {{ text: 'Gradient Stability', font: {{ size: 14, color: '#f3f4f6' }} }},
                yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Norm', type: 'log' }}),
                xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Step' }}),
                showlegend: false,
            }}), plotCfg);
        }}

        // ── 4. Perplexity (if available) ────────────────────────
        if (document.getElementById('chart-ppl')) {{
            const pplTraces = [createTrace('train/perplexity', 'Perplexity', C.violet)].filter(t => t);
            if (pplTraces.length > 0) {{
                Plotly.newPlot('chart-ppl', pplTraces, Object.assign({{}}, baseLayout, {{
                    title: {{ text: 'Perplexity (exp MLM Loss)', font: {{ size: 14, color: '#f3f4f6' }} }},
                    yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Perplexity', type: 'log' }}),
                    xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Step' }}),
                    showlegend: false,
                }}), plotCfg);
            }}
        }}

        // ── 5. Learning Rate Schedule (if available) ────────────
        if (document.getElementById('chart-lr')) {{
            const lrTraces = [
                createTrace('train/lr_encoder', 'Encoder LR', C.sky),
                createTrace('train/lr_wire',    'WIRE/RoPE LR', C.violet),
                createTrace('train/lr_rope',    'RoPE LR',     C.violet),
            ].filter(t => t);
            if (lrTraces.length > 0) {{
                Plotly.newPlot('chart-lr', lrTraces, Object.assign({{}}, baseLayout, {{
                    title: {{ text: 'Learning Rate Schedule', font: {{ size: 14, color: '#f3f4f6' }} }},
                    yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'LR', type: 'log' }}),
                    xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Step' }}),
                }}), plotCfg);
            }}
        }}

        // ── 6. RoPE Feature Utilization (if available) ──────────
        if (document.getElementById('chart-util')) {{
            const utilKeys = Object.keys(metrics).filter(k => k.startsWith('rot/util_'));
            if (utilKeys.length > 0) {{
                const utilLabels = utilKeys.map(k => k.replace('rot/util_', ''));
                const utilVals = utilKeys.map(k => getLast(k) || 0);
                const barColors = [C.teal, C.sky, C.indigo, C.violet, C.amber, C.rose];
                Plotly.newPlot('chart-util', [{{
                    x: utilLabels, y: utilVals, type: 'bar',
                    marker: {{ color: utilLabels.map((_, i) => barColors[i % barColors.length]), opacity: 0.85 }},
                    hovertemplate: '%{{x}}: %{{y:.3f}}<extra></extra>',
                }}], Object.assign({{}}, baseLayout, {{
                    title: {{ text: 'RoPE Feature Utilization', font: {{ size: 14, color: '#f3f4f6' }} }},
                    yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Fraction', range: [0, 1] }}),
                    showlegend: false, bargap: 0.35,
                }}), plotCfg);
            }}
        }}

        // ── 7. Entropy Heatmap (if available) ───────────────────
        if (document.getElementById('chart-entropy')) {{
            const entropyKeys = Object.keys(metrics).filter(k => k.startsWith('xai/entropy_layer'));
            if (entropyKeys.length > 0) {{
                let maxL = 0, maxH = 0;
                entropyKeys.forEach(k => {{
                    const m = k.match(/layer(\\d+)_head(\\d+)/);
                    if (m) {{ maxL = Math.max(maxL, parseInt(m[1])); maxH = Math.max(maxH, parseInt(m[2])); }}
                }});
                const z = Array(maxL + 1).fill().map(() => Array(maxH + 1).fill(0));
                entropyKeys.forEach(k => {{
                    const m = k.match(/layer(\\d+)_head(\\d+)/);
                    if (m) {{ z[parseInt(m[1])][parseInt(m[2])] = getLast(k) || 0; }}
                }});
                Plotly.newPlot('chart-entropy', [{{
                    z: z, type: 'heatmap',
                    colorscale: [[0,'#111827'],[0.25,'#1e3a5f'],[0.5,'#38bdf8'],[0.75,'#a78bfa'],[1,'#fb7185']],
                    colorbar: {{ title: 'Entropy', tickfont: {{ color: '#9ca3af' }}, titlefont: {{ color: '#9ca3af' }} }},
                    hovertemplate: 'Layer %{{y}}, Head %{{x}}: %{{z:.3f}}<extra></extra>',
                }}], Object.assign({{}}, baseLayout, {{
                    title: {{ text: 'Attention Entropy Heatmap', font: {{ size: 14, color: '#f3f4f6' }} }},
                    xaxis: Object.assign({{}}, baseLayout.xaxis, {{ title: 'Head Index' }}),
                    yaxis: Object.assign({{}}, baseLayout.yaxis, {{ title: 'Layer Index' }}),
                }}), plotCfg);
            }}
        }}

        // ── Populate Stat Cards ──────────────────────────────
        const fmt = (v, d) => v !== null ? v.toFixed(d) : '\u2014';

        // Best val loss (find minimum across all val/loss values)
        const valLossArr = metrics['val/loss'] || [];
        const valLossNonNull = valLossArr.filter(v => v !== null && v !== undefined);
        const bestValLoss = valLossNonNull.length > 0 ? Math.min(...valLossNonNull) : null;
        document.getElementById('stat-val-loss').innerText = fmt(bestValLoss, 4);

        document.getElementById('stat-loss').innerText = fmt(getLast('train/loss'), 4);

        const mlmV = getLast('train/mlm_acc');
        document.getElementById('stat-mlm').innerText = mlmV !== null ? (mlmV * 100).toFixed(1) + '%' : '\u2014';

        // Perplexity or val MLM acc
        const pplEl = document.getElementById('stat-ppl');
        if (pplEl) {{
            {'const pplV = getLast("train/perplexity"); pplEl.innerText = pplV !== null ? pplV.toFixed(1) : "—";' if has_perplexity else 'const vma = getLast("val/mlm_acc"); pplEl.innerText = vma !== null ? (vma * 100).toFixed(1) + "%" : "—";'}
        }}

        document.getElementById('stat-err').innerText = fmt(getLast('train/err_loss'), 4);
        document.getElementById('stat-grad').innerText = fmt(getLast('train/grad_norm'), 2);

        const gpuV = getLast('train/gpu_mem_gb');
        document.getElementById('stat-gpu').innerText = gpuV !== null ? gpuV.toFixed(1) + ' GB' : '\u2014';

        // Training time or val MLM acc
        const timeEl = document.getElementById('stat-time');
        if (timeEl) {{
            {'const wt = getLast("train/wall_time_min"); timeEl.innerText = wt !== null ? (wt/60).toFixed(1) + "h" : "—";' if has_wall_time else 'const vma2 = getLast("val/mlm_acc"); timeEl.innerText = vma2 !== null ? (vma2 * 100).toFixed(1) + "%" : "—";'}
        }}

        // ── Alert List ───────────────────────────────────────
        const alertList = document.getElementById('alert-list');
        if (alertList) {{
            alerts.slice().reverse().forEach(a => {{
                const div = document.createElement('div');
                div.className = `alert-item level-${{a.level}}`;
                div.innerHTML = `<span class="alert-ts">${{a.timestamp.slice(11)}}</span> ${{a.message}}`;
                alertList.appendChild(div);
            }});
        }}
    </script>
</body>
</html>"""
        dashboard_path = self.log_dir / "dashboard.html"
        with open(dashboard_path, "w") as f: f.write(html)
        self.log_alert(f"Dashboard updated: {dashboard_path}", "INFO")

    def run(self):
        print(f"{GREEN}Sentry active.{RESET} Logs -> {self.alert_path}")
        self.log_alert(f"Sentry started. Monitoring {self.log_dir}", "INFO")
        
        last_data_check = 0
        last_dashboard = 0
        while True:
            self.check_training_metrics()
            
            now = time.time()
            if now - last_data_check > self.data_interval:
                self.check_dataset_health()
                last_data_check = now

            # Regenerate dashboard every 30 seconds
            if now - last_dashboard > 30:
                self.generate_dashboard()
                last_dashboard = now
            
            time.sleep(self.log_interval)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config")
    parser.add_argument("--log-dir", help="Path to logs")
    parser.add_argument("--feat-dir", help="Path to features")
    parser.add_argument("--log-interval", type=int, default=60)
    parser.add_argument("--data-interval", type=int, default=21600)
    
    args = parser.parse_args()

    # Apply config
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        if "paths" in cfg:
            if not args.log_dir: args.log_dir = cfg["paths"].get("log_dir")
            if not args.feat_dir: args.feat_dir = cfg["paths"].get("feat_dir")
        if "logging" in cfg:
            if args.log_interval == 60:
                args.log_interval = cfg["logging"].get("sentry_log_interval", 60)
            if args.data_interval == 21600:
                args.data_interval = cfg["logging"].get("sentry_data_interval", 21600)

    if not args.log_dir or not args.feat_dir:
        print("Error: Paths required")
        sys.exit(1)

    sentry = Sentry(args.log_dir, args.feat_dir, args.log_interval, args.data_interval)
    sentry.run()
