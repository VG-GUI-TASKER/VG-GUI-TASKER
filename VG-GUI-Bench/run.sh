#!/usr/bin/env bash
# Usage: bash run.sh <mode> <cut|nocut> [num_threads]
# Examples:
#   bash run.sh single cut
#   bash run.sh origin nocut
#   bash run.sh uniform10 cut 32
#
# Available modes: single / origin / gt / annotation / tasker / bfs / gbfs /
#                  dijkstra / videoagent / videotree / uniform5 / uniform10
#
# The model is any OpenAI-compatible VLM endpoint. Configure it via environment
# variables (or pass --model_name / --api_key / --base_url explicitly):
#
#   # Official OpenAI
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_MODEL="gpt-4o"
#
#   # A local vLLM server hosting an open-source VLM
#   export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
#   export OPENAI_API_KEY="EMPTY"
#   export OPENAI_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

set -e

REF_MODE="${1:?Usage: bash run.sh <mode> <cut|nocut> [num_threads]}"
CUT_FLAG="${2:?Please specify cut or nocut}"
NUM_THREADS="${3:-16}"

if [[ "$CUT_FLAG" != "cut" && "$CUT_FLAG" != "nocut" ]]; then
    echo "ERROR: the second argument must be 'cut' or 'nocut', got: '$CUT_FLAG'"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Dataset root (override with the DATASET_ROOT env var if needed)
DATASET_ROOT="${DATASET_ROOT:-./MONDAY}"

NOCUT_ARG=""
if [[ "$CUT_FLAG" == "nocut" ]]; then
    NOCUT_ARG="--no_cut"
fi

echo "=== Running eval with ref_mode=${REF_MODE}, ${CUT_FLAG}, threads=${NUM_THREADS} ==="

python -m core.eval_qwen \
  --max_tokens 8192 \
  --temperature 0.6 \
  --test_json_path "${DATASET_ROOT}/ours_data.json" \
  --dataset_root "${DATASET_ROOT}" \
  --ref_mode "$REF_MODE" \
  --log_root ./logs/ \
  --task monday \
  --num_history 1 \
  --num_threads "$NUM_THREADS" \
  $NOCUT_ARG
