#!/usr/bin/env bash
# 用法: bash run.sh <mode> <cut|nocut> [num_threads]
# 例如: bash run.sh single cut
#       bash run.sh origin nocut
#       bash run.sh uniform10 cut 32
#
# 可选 mode: single / origin / gt / annotation / tasker / bfs / gbfs / dijkstra / videoagent / videotree / uniform5 / uniform10

REF_MODE="${1:?用法: bash run.sh <mode> <cut|nocut> [num_threads]}"
CUT_FLAG="${2:?请指定 cut 或 nocut}"
NUM_THREADS="${3:-16}"

# 校验 cut/nocut
if [[ "$CUT_FLAG" != "cut" && "$CUT_FLAG" != "nocut" ]]; then
    echo "ERROR: 第二个参数必须是 'cut' 或 'nocut'，收到: '$CUT_FLAG'"
    exit 1
fi

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME:${PYTHONPATH:-}"
export PYTHONPATH="$PYTHONPATH:$(pwd)/api"

# 组装 --no_cut 参数
NOCUT_ARG=""
if [[ "$CUT_FLAG" == "nocut" ]]; then
    NOCUT_ARG="--no_cut"
fi

echo "=== Running eval_qwen with ref_mode=${REF_MODE}, ${CUT_FLAG}, threads=${NUM_THREADS} ==="

CUDA_VISIBLE_DEVICES=0 python -m core.eval_qwen \
  --use_api \
  --model_type qwen35 \
  --qwen_name_list all \
  --max_tokens 8192 \
  --temperature 0.6 \
  --test_json_path /data/home/stevefan/projects/lql/VG-GUI-Bench/MONDAY/ours_data.json \
  --dataset_root /data/home/stevefan/projects/lql/VG-GUI-Bench/MONDAY \
  --ref_mode "$REF_MODE" \
  --log_root ./logs/ \
  --task monday \
  --num_history 1 \
  --num_threads "$NUM_THREADS" \
  $NOCUT_ARG
