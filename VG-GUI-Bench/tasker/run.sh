#!/bin/bash

# Usage:
#   bash run.sh         # 跑全部数据
#   bash run.sh 1       # 跑第1个shard (0%-10%)
#   bash run.sh 5       # 跑第5个shard (40%-50%)
#
# 10个tmux窗口并行：
#   for i in $(seq 1 10); do tmux new-window -t newakeys$i "cd /path/to/akeys && bash run.sh $i"; done

SHARD=${1:-0}  # 默认0=跑全部

# 1. 动态获取各级绝对路径
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$CURRENT_DIR")"

# 2. 把 项目根目录、qwen目录、以及你的家目录 ~ 全部加入 PYTHONPATH！
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/qwen:$HOME:${PYTHONPATH}"

# 3. 强制切回到你跑 eval_qwen_annot 的主场目录
cd $HOME

echo "Project Root: $PROJECT_ROOT"
echo "Working Directory: $(pwd)"
echo "Running AKeyS extraction (shard=$SHARD)..."

# 4. 路径配置
VIDEO_DIR="/data/home/stevefan/projects/monday/ytb_video_test"
JSON_DIR="/data/home/stevefan/projects/monday/ours_data_test.json"
OUT_ROOT="/data/home/stevefan/projects/monday/images/test_tasker1d_new"
CACHE_DIR="/tmp/akeys_frame_cache_tasker1d_new"
RECORD_JSON_PATH="/data/home/stevefan/projects/monday/akeys_selected_frames_tasker1d_new.json"

# VIDEO_DIR="/root/projects/monday/ytb_video_test"
# JSON_DIR="/root/projects/monday/ours_data_test.json"
# OUT_ROOT="/root/projects/monday/images/test_tasker1g"
# CACHE_DIR="/tmp/akeys_frame_cache_tasker1g"
# RECORD_JSON_PATH="/root/projects/monday/akeys_selected_frames_tasker1g.json"


# 5. 使用绝对路径跨目录调用 main.py
python $CURRENT_DIR/main.py \
    --video_dir $VIDEO_DIR \
    --json_dir $JSON_DIR \
    --out_root $OUT_ROOT \
    --cache_dir $CACHE_DIR \
    --record_json_path $RECORD_JSON_PATH \
    --search_strategy dijkstra \
    --init_interval 4 \
    --final_step 6 \
    --min_steps 3 \
    --beam_size 1 \
    --conf_lower 3 \
    --temperature 0.0 \
    --qwen_name_list all \
    --max_workers 16 \
    --shard $SHARD
