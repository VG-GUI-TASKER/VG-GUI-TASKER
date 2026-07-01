#!/bin/bash
# TASKER (GUI stage) keyframe extraction on GUI tutorial videos.
#
# Usage:
#   bash run.sh          # process all data
#   bash run.sh 1        # process shard 1 (0%-10%)
#   bash run.sh 5        # process shard 5 (40%-50%)
#
# Run 10 shards in parallel with tmux:
#   for i in $(seq 1 10); do tmux new-window -t tasker$i "bash run.sh $i"; done
#
# The model is any OpenAI-compatible VLM endpoint. Configure it via env vars:
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_MODEL="gpt-4o"
#   # optional, for a local / self-hosted server:
#   # export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

set -e

SHARD=${1:-0}  # 0 = process all

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${CURRENT_DIR}:${PYTHONPATH:-}"

# ---- Paths (override with env vars or edit here) ----
# Defaults assume a MONDAY dataset laid out next to the benchmark.
DATASET_ROOT="${DATASET_ROOT:-./MONDAY}"
VIDEO_DIR="${VIDEO_DIR:-${DATASET_ROOT}/ytb_video_test}"
JSON_DIR="${JSON_DIR:-${DATASET_ROOT}/ours_data_test.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_ROOT}/images/tasker}"
CACHE_DIR="${CACHE_DIR:-/tmp/tasker_frame_cache}"
RECORD_JSON_PATH="${RECORD_JSON_PATH:-${DATASET_ROOT}/tasker_selected_frames.json}"

echo "Working Directory: $(pwd)"
echo "Running TASKER extraction (shard=$SHARD)..."

python "$CURRENT_DIR/main.py" \
    --video_dir "$VIDEO_DIR" \
    --json_dir "$JSON_DIR" \
    --out_root "$OUT_ROOT" \
    --cache_dir "$CACHE_DIR" \
    --record_json_path "$RECORD_JSON_PATH" \
    --search_strategy dijkstra \
    --init_interval 4 \
    --final_step 6 \
    --min_steps 3 \
    --beam_size 1 \
    --conf_lower 3 \
    --temperature 0.0 \
    --max_workers 16 \
    --shard "$SHARD"
