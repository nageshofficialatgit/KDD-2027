#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  run_ablations.sh — Launch all ablation experiments in detached screens
#
#  Usage:
#    bash experiments/run_ablations.sh                  # Run all
#    bash experiments/run_ablations.sh config_e1_s1_full.json  # Run one
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs"
TRAIN_SCRIPT="$SCRIPT_DIR/train_graph_rope.py"

# If specific config name given, use it; otherwise run all ablation configs
if [[ $# -ge 1 ]]; then
    CONFIGS=("$@")
else
    CONFIGS=(
        "config_e1_s1_full.json"
        "config_e1_s1_mlm_only.json"
        "config_e1_s1_mlm_tfr.json"
        "config_e1_s1_mlm_edge.json"
    )
fi

echo "═══════════════════════════════════════════════════════"
echo "  Graph-RoPE Ablation Launcher"
echo "═══════════════════════════════════════════════════════"
echo ""

for cfg_name in "${CONFIGS[@]}"; do
    cfg_path="$CONFIG_DIR/$cfg_name"
    if [[ ! -f "$cfg_path" ]]; then
        echo "⚠️  Config not found: $cfg_path — skipping"
        continue
    fi

    # Derive screen session name from config (e.g. config_e1_s1_full.json → e1_s1_full)
    session_name=$(basename "$cfg_name" .json | sed 's/^config_//')

    # Check if session already exists
    if screen -list | grep -q "$session_name"; then
        echo "⚠️  Screen '$session_name' already running — skipping"
        continue
    fi

    echo "🚀 Launching: $session_name"
    echo "   Config:  $cfg_path"
    echo "   Screen:  $session_name"

    screen -dmS "$session_name" bash -c "
        cd $PROJECT_DIR && \
        python $TRAIN_SCRIPT --config $cfg_path 2>&1 | tee logs/${session_name}.log
    "
    echo "   ✅ Started"
    echo ""
done

echo "═══════════════════════════════════════════════════════"
echo "  Active sessions:"
screen -list 2>/dev/null | grep -E "($(IFS='|'; echo "${CONFIGS[*]}" | sed 's/config_//g;s/\.json//g'))" || echo "  (none matched)"
echo ""
echo "  Attach: screen -r <session_name>"
echo "  Detach: Ctrl+A, D"
echo "═══════════════════════════════════════════════════════"
