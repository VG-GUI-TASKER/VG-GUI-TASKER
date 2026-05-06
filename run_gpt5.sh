#!/usr/bin/env bash
# ============================================================================
# VG-GUI-Bench GPT-5 系列评测运行脚本
# ============================================================================
# 用法:
#   bash run_gpt5.sh [all|single|uniform10|gpt5|gpt5_mini]
#
# 模型列表: gpt5, gpt5_mini
# 模式列表: single (Mode A), uniform10 (Mode B)
#
# 示例:
#   bash run_gpt5.sh all          # 运行全部 4 组评测 (2模型 × 2模式)
#   bash run_gpt5.sh single       # 运行全部模型的 single 模式
#   bash run_gpt5.sh gpt5         # 运行 GPT-5 的两种模式
#   bash run_gpt5.sh gpt5_mini    # 运行 GPT-5-Mini 的两种模式
#
# 前置要求:
#   1. 设置环境变量 AIPING_API_KEY 或修改 api/aiping_adapter.py 中的 AIPING_API_KEY
#   2. 确保 MONDAY 数据集已在 DATASET_ROOT 路径下
# ============================================================================

set -e

# ==================== 配置区 ====================
PROJECT_DIR="/root/projects/lql/VG-GUI-Bench"
TEST_JSON="${PROJECT_DIR}/MONDAY/ours_data.json"
DATASET_ROOT="${PROJECT_DIR}/MONDAY"
LOG_ROOT="${PROJECT_DIR}/logs"
TMUX_SESSION="vg_bench_gpt5"

# API Key (也可以通过环境变量 AIPING_API_KEY 设置)
# AIPING_API_KEY="YOUR_API_KEY_HERE"  # <-- 取消注释并填入你的 Key

# 线程数 (aiping.cn 需要根据你的账户限频调整, 建议从 4 开始)
THREADS=4

# 所有模型
ALL_MODELS=("gpt5" "gpt5_mini")
# 所有模式
ALL_MODES=("single" "uniform10")

# ==================== 环境变量 ====================
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME:${PYTHONPATH:-}"
export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/api"

# 如果上面设置了 AIPING_API_KEY，导出为环境变量
if [ -n "${AIPING_API_KEY:-}" ]; then
    export AIPING_API_KEY
fi

# ==================== 辅助函数 ====================

run_single_eval() {
    local model=$1
    local mode=$2
    local eval_name="${model}_${mode}"

    echo "[$(date '+%H:%M:%S')] 启动: model=${model}, mode=${mode}, threads=${THREADS}"

    cd "$PROJECT_DIR"

    python -m core.eval_qwen \
        --use_api \
        --model_type "$model" \
        --qwen_name_list all \
        --max_tokens 8192 \
        --temperature 0.6 \
        --test_json_path "$TEST_JSON" \
        --dataset_root "$DATASET_ROOT" \
        --ref_mode "$mode" \
        --log_root "${LOG_ROOT}/${eval_name}/" \
        --eval_name "$eval_name" \
        --task monday \
        --num_history 1 \
        --num_threads "$THREADS"

    echo "[$(date '+%H:%M:%S')] 完成: ${eval_name}"
}

# ==================== 主逻辑 ====================

MODE_FILTER="${1:-all}"

# 确定需要运行的模型和模式
declare -a RUN_MODELS
declare -a RUN_MODES

case "$MODE_FILTER" in
    all)
        RUN_MODELS=("${ALL_MODELS[@]}")
        RUN_MODES=("${ALL_MODES[@]}")
        ;;
    single)
        RUN_MODELS=("${ALL_MODELS[@]}")
        RUN_MODES=("single")
        ;;
    uniform10)
        RUN_MODELS=("${ALL_MODELS[@]}")
        RUN_MODES=("uniform10")
        ;;
    *)
        # 假设是模型名
        found=false
        for m in "${ALL_MODELS[@]}"; do
            if [[ "$m" == "$MODE_FILTER" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" == "true" ]]; then
            RUN_MODELS=("$MODE_FILTER")
            RUN_MODES=("${ALL_MODES[@]}")
        else
            echo "ERROR: 未知参数 '$MODE_FILTER'"
            echo "用法: bash run_gpt5.sh [all|single|uniform10|模型名]"
            echo "模型名: ${ALL_MODELS[*]}"
            exit 1
        fi
        ;;
esac

echo "============================================================"
echo " VG-GUI-Bench GPT-5 系列评测"
echo "============================================================"
echo " 模型: ${RUN_MODELS[*]}"
echo " 模式: ${RUN_MODES[*]}"
echo " 总任务数: $(( ${#RUN_MODELS[@]} * ${#RUN_MODES[@]} ))"
echo " 线程数: ${THREADS}"
echo "============================================================"

# 创建日志目录
mkdir -p "$LOG_ROOT"

# 检查是否使用 tmux 并行
if command -v tmux &> /dev/null && [[ $(( ${#RUN_MODELS[@]} * ${#RUN_MODES[@]} )) -gt 1 ]]; then
    echo ""
    echo ">>> 使用 tmux 并行执行（session: $TMUX_SESSION）"
    echo ""

    # 如果 session 已存在则复用，否则新建
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "  (复用已有 tmux session: $TMUX_SESSION)"
    else
        tmux new-session -d -s "$TMUX_SESSION" -n "control"
    fi
    
    # 为每个任务创建一个窗口
    for model in "${RUN_MODELS[@]}"; do
        for mode in "${RUN_MODES[@]}"; do
            local_eval_name="${model}_${mode}"
            
            # 构建运行命令
            CMD="conda activate guirl_lai_sunqi && "
            CMD+="cd ${PROJECT_DIR} && "
            CMD+="export PYTHONUNBUFFERED=1 && "
            CMD+="export PYTHONPATH=\"\$HOME:\${PYTHONPATH:-}\" && "
            CMD+="export PYTHONPATH=\"\$PYTHONPATH:${PROJECT_DIR}/api\" && "
            # API Key (取消注释并填入)
            # CMD+="export AIPING_API_KEY='YOUR_API_KEY_HERE' && "
            CMD+="mkdir -p ${LOG_ROOT}/${local_eval_name}/ && "
            CMD+="python -m core.eval_qwen "
            CMD+="--use_api --model_type ${model} --qwen_name_list all "
            CMD+="--max_tokens 8192 --temperature 0.6 "
            CMD+="--test_json_path ${TEST_JSON} --dataset_root ${DATASET_ROOT} "
            CMD+="--ref_mode ${mode} --log_root ${LOG_ROOT}/${local_eval_name}/ "
            CMD+="--eval_name ${local_eval_name} --task monday --num_history 1 "
            CMD+="--num_threads ${THREADS} "
            CMD+="2>&1 | tee ${LOG_ROOT}/${local_eval_name}/run.log"
            
            tmux new-window -t "$TMUX_SESSION" -n "$local_eval_name"
            tmux send-keys -t "$TMUX_SESSION:${local_eval_name}" "$CMD" C-m
            
            echo "  [tmux] 窗口 '${local_eval_name}' 已启动"
        done
    done

    echo ""
    echo "============================================================"
    echo " 全部任务已在 tmux 中启动！"
    echo " 查看: tmux attach -t $TMUX_SESSION"
    echo " 切换窗口: Ctrl+B + n/p 或 Ctrl+B + 窗口编号"
    echo "============================================================"

else
    # 无 tmux 或单个任务，串行执行
    echo ">>> 串行执行"
    for model in "${RUN_MODELS[@]}"; do
        for mode in "${RUN_MODES[@]}"; do
            mkdir -p "${LOG_ROOT}/${model}_${mode}/"
            run_single_eval "$model" "$mode"
        done
    done
    echo ""
    echo "全部评测完成！请运行 python aggregate_leaderboard.py 汇总结果。"
fi
