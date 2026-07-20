#!/bin/bash
# ============================================================
# ERR Computation — Screen Launcher
# ============================================================
# Runs compute_err.py for each bucket in a detached screen.
# Auto-resumes if checkpoints exist.
#
# Usage: bash run_err_compute.sh
# To check progress: screen -r err_compute
# To detach: Ctrl+A, D
# ============================================================

set -e

PROJECT_DIR=os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
DATA_PREP="${PROJECT_DIR}/Implementation/data_prep"
JSONL_DIR="${PROJECT_DIR}/Dataset_for_first/cleaned_buckets"
OUTPUT_DIR="${PROJECT_DIR}/model/cpg_features"
VENV="${PROJECT_DIR}/.venv/bin/activate"

SESSION_NAME="err_compute"
LOG_FILE="${PROJECT_DIR}/logs/err_compute.log"
WRAPPER="${PROJECT_DIR}/scripts/.err_wrapper.sh"

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

# Kill existing session if any
screen -S "$SESSION_NAME" -X quit 2>/dev/null || true

# --- CLEANUP (disabled for resume) ---
# echo "Cleaning old ERR data..."
# rm -f "${OUTPUT_DIR}"/err_*.npz
# rm -f "${OUTPUT_DIR}"/err_*_checkpoint.json
# rm -rf "${OUTPUT_DIR}"/.tmp_err_*
# rm -f "${LOG_FILE}"
# echo "  Deleted old err_*.npz, checkpoints, and tmp dirs"

# Write the wrapper script (avoids heredoc escaping issues)
cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
source .venv/bin/activate
cd ./data_prep

LOG="./logs/err_compute.log"
JSONL="./Dataset_for_first/cleaned_buckets"
OUT="./model/cpg_features"

echo "============================================================" | tee -a "$LOG"
echo "ERR Computation — Started at $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Process ALL buckets
for BUCKET in v0.1 v0.3 v0.4 v0.5 v0.6 v0.7 v0.8 v0.9 v6.0 v8.7; do
    JSONL_FILE="${JSONL}/${BUCKET}.jsonl"
    if [ ! -f "$JSONL_FILE" ]; then
        echo ">>> SKIP ${BUCKET} (file not found)" | tee -a "$LOG"
        continue
    fi
    echo "" | tee -a "$LOG"
    echo ">>> Processing ${BUCKET}..." | tee -a "$LOG"
    python compute_err.py \
        --jsonl "$JSONL_FILE" \
        --output-dir "$OUT" \
        --pairs-per-contract 128 \
        --workers 48 \
        2>&1 | tee -a "$LOG"
    echo ">>> ${BUCKET} complete at $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "ERR Computation — ALL BUCKETS FINISHED at $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Keep screen alive for inspection
echo "Done. Press Enter to close screen."
read
WRAPPER_EOF

chmod +x "$WRAPPER"

# Launch in detached screen
screen -dmS "$SESSION_NAME" bash "$WRAPPER"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ERR Compute launched in screen: ${SESSION_NAME}        ║"
echo "║                                                          ║"
echo "║  Check progress:  screen -r ${SESSION_NAME}             ║"
echo "║  Detach:           Ctrl+A, D                             ║"
echo "║  Log file:         logs/err_compute.log                  ║"
echo "║                                                          ║"
echo "║  Buckets: ALL (v0.1-v0.9, v6.0, v8.7)                   ║"
echo "║  K=128 pairs/contract (RW-stratified), 24 workers          ║"
echo "╚══════════════════════════════════════════════════════════╝"
