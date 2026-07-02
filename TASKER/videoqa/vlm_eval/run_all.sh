#!/bin/bash
# ============================================================
# VLM-based VideoQA evaluation runner
# Evaluates keyframe-selection methods on EgoSchema and NExT-QA.
# ============================================================
# Usage:
#   bash run_all.sh                       # run all methods x both datasets
#   bash run_all.sh --method uniform      # only the Uniform baseline
#   bash run_all.sh --dataset egoschema   # only EgoSchema
#   bash run_all.sh --method tasker --dataset nextqa
#
# The model endpoint is configured via environment variables:
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_MODEL="gpt-4o-2024-11-20"        # or Qwen3-VL-235B-A22B-Instruct, ...
#   export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"   # optional
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============ Defaults ============
METHOD="all"             # textonly / uniform / videotree / videoagent / tasker / all
DATASET="both"           # egoschema / nextqa / both
EGOSCHEMA_SPLIT="subset" # subset (500, has ground truth) / full (5031, submit to server) / both
NUM_FRAMES=16            # Uniform sampling frames
MAX_WORKERS=16           # parallel requests (Uniform / TASKER / VideoAgent)
VIDEOTREE_WORKERS=8      # VideoTree parallelism (CLIP feature extraction)
CACHE_DIR="/tmp/benchmark_frames"

# ============ Parse args ============
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --egoschema_split) EGOSCHEMA_SPLIT="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        --max_workers) MAX_WORKERS="$2"; shift 2 ;;
        --videotree_workers) VIDEOTREE_WORKERS="$2"; shift 2 ;;
        --cache_dir) CACHE_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD          textonly|uniform|videotree|videoagent|tasker|all (default: all)"
            echo "  --dataset DATASET        egoschema|nextqa|both (default: both)"
            echo "  --egoschema_split SPLIT  subset|full|both (default: subset)"
            echo "  --num_frames N           Uniform sampling frames (default: 16)"
            echo "  --max_workers N          Max parallel workers (default: 16)"
            echo "  --videotree_workers N    VideoTree parallel workers (default: 8)"
            echo "  --cache_dir DIR          Frame cache directory (default: /tmp/benchmark_frames)"
            echo ""
            echo "Table 1 metrics:"
            echo "  EgoSchema: Sub. (subset 500), Full (full 5031)"
            echo "  NExT-QA:   Tem. (Temporal), Cau. (Causal), Des. (Descriptive), Avg."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${OPENAI_MODEL}" ]]; then
    echo "  [WARNING] OPENAI_MODEL is not set. Set OPENAI_API_KEY / OPENAI_MODEL"
    echo "            (and optionally OPENAI_BASE_URL) before running." >&2
fi

# ============ Runners ============
run_textonly() {
    echo "== [Text-only] blind baseline (no frames) | dataset: $1 =="
    python "${SCRIPT_DIR}/eval_textonly.py" \
        --dataset "$1" --max_workers ${MAX_WORKERS} --egoschema_split "${EGOSCHEMA_SPLIT}"
}

run_uniform() {
    echo "== [Uniform] ${NUM_FRAMES} frames | dataset: $1 =="
    python "${SCRIPT_DIR}/eval_uniform.py" \
        --dataset "$1" --num_frames ${NUM_FRAMES} --max_workers ${MAX_WORKERS} \
        --cache_dir "${CACHE_DIR}" --egoschema_split "${EGOSCHEMA_SPLIT}"
}

run_videotree() {
    echo "== [VideoTree] CLIP ViT-L/14 adaptive selection | dataset: $1 =="
    python "${SCRIPT_DIR}/eval_videotree.py" \
        --dataset "$1" --max_workers ${VIDEOTREE_WORKERS} \
        --cache_dir "${CACHE_DIR}" --egoschema_split "${EGOSCHEMA_SPLIT}"
}

run_videoagent() {
    echo "== [VideoAgent] VLM-guided iterative selection (ECCV 2024) | dataset: $1 =="
    python "${SCRIPT_DIR}/eval_videoagent.py" \
        --dataset "$1" --max_workers ${MAX_WORKERS} \
        --max_frames 16 --num_init_frames 5 --max_iterations 2 \
        --cache_dir "${CACHE_DIR}" --egoschema_split "${EGOSCHEMA_SPLIT}"
}

run_tasker() {
    echo "== [TASKER] coverage-aware A* keyframe search -> VLM QA (ours) | dataset: $1 =="
    python "${SCRIPT_DIR}/eval_tasker.py" \
        --dataset "$1" --max_workers ${MAX_WORKERS} \
        --search_strategy a_star --max_frames 16 --init_frames 4 \
        --conf_lower 3 --min_steps 2 \
        --cache_dir "${CACHE_DIR}" --egoschema_split "${EGOSCHEMA_SPLIT}"
}

echo "============================================================"
echo "  VideoQA evaluation | method=${METHOD} dataset=${DATASET} egoschema_split=${EGOSCHEMA_SPLIT}"
echo "  model=${OPENAI_MODEL:-<unset>}"
echo "============================================================"

if [[ "${METHOD}" == "textonly"   || "${METHOD}" == "all" ]]; then run_textonly   "${DATASET}"; fi
if [[ "${METHOD}" == "uniform"    || "${METHOD}" == "all" ]]; then run_uniform    "${DATASET}"; fi
if [[ "${METHOD}" == "videotree"  || "${METHOD}" == "all" ]]; then run_videotree  "${DATASET}"; fi
if [[ "${METHOD}" == "videoagent" || "${METHOD}" == "all" ]]; then run_videoagent "${DATASET}"; fi
if [[ "${METHOD}" == "tasker"     || "${METHOD}" == "all" ]]; then run_tasker     "${DATASET}"; fi

echo ""
echo "Done. Results are under: ${VIDEOQA_RESULTS_DIR:-${SCRIPT_DIR}/results}"
