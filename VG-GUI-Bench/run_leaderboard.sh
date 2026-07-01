#!/usr/bin/env bash
# ============================================================================
# VG-GUI-Bench Leaderboard evaluation runner
# ============================================================================
# Usage:
#   bash run_leaderboard.sh [all|single|uniform10] [eval_name]
#
# Evaluation protocols:
#   single    (Mode A): target screen only, no video context
#   uniform10 (Mode B): 10 uniformly-sampled reference frames + target screen
#
# The model under evaluation is any OpenAI-compatible VLM endpoint, configured
# through environment variables:
#
#   # Official OpenAI
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_MODEL="gpt-4o"
#
#   # A local vLLM server hosting an open-source VLM
#   export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
#   export OPENAI_API_KEY="EMPTY"
#   export OPENAI_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
#
# Examples:
#   export OPENAI_MODEL=gpt-4o
#   bash run_leaderboard.sh all gpt-4o        # run both modes for gpt-4o
#   bash run_leaderboard.sh single gpt-4o     # only Mode A
#   bash run_leaderboard.sh uniform10 gpt-4o  # only Mode B
# ============================================================================

set -e

# ==================== Configuration ====================
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/MONDAY}"
TEST_JSON="${DATASET_ROOT}/ours_data.json"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logs}"

# Number of concurrent requests (tune to your endpoint's rate limits)
NUM_THREADS="${NUM_THREADS:-16}"

ALL_MODES=("single" "uniform10")

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ==================== Argument parsing ====================
MODE_FILTER="${1:-all}"
EVAL_TAG="${2:-${OPENAI_MODEL:-model}}"
# sanitise eval tag for use in file paths
EVAL_TAG="$(echo "$EVAL_TAG" | tr '/ :' '___')"

declare -a RUN_MODES
case "$MODE_FILTER" in
    all)        RUN_MODES=("${ALL_MODES[@]}") ;;
    single)     RUN_MODES=("single") ;;
    uniform10)  RUN_MODES=("uniform10") ;;
    *)
        echo "ERROR: unknown argument '$MODE_FILTER'"
        echo "Usage: bash run_leaderboard.sh [all|single|uniform10] [eval_name]"
        exit 1
        ;;
esac

echo "============================================================"
echo " VG-GUI-Bench Leaderboard evaluation"
echo "============================================================"
echo " Model : ${OPENAI_MODEL:-<from --model_name>}"
echo " Modes : ${RUN_MODES[*]}"
echo " Tag   : ${EVAL_TAG}"
echo "============================================================"

mkdir -p "$LOG_ROOT"

run_single_eval() {
    local mode=$1
    local eval_name="${EVAL_TAG}_${mode}"
    echo "[$(date '+%H:%M:%S')] start: mode=${mode}"
    mkdir -p "${LOG_ROOT}/${eval_name}/"
    cd "$PROJECT_DIR"
    python -m core.eval_qwen \
        --max_tokens 8192 \
        --temperature 0.6 \
        --test_json_path "$TEST_JSON" \
        --dataset_root "$DATASET_ROOT" \
        --ref_mode "$mode" \
        --log_root "${LOG_ROOT}/${eval_name}/" \
        --eval_name "$eval_name" \
        --task monday \
        --num_history 1 \
        --num_threads "$NUM_THREADS"
    echo "[$(date '+%H:%M:%S')] done: ${eval_name}"
}

for mode in "${RUN_MODES[@]}"; do
    run_single_eval "$mode"
done

echo ""
echo "All evaluations finished. Run 'python aggregate_leaderboard.py' to build the leaderboard."
