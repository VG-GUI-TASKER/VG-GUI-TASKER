#!/usr/bin/env bash
# ============================================================================
# VG-GUI-Bench Leaderboard 评测统一运行脚本
# ============================================================================
# 用法:
#   bash run_leaderboard.sh [all|single|uniform10|模型名]
#
# 模型列表: qwen3vl, gemini_flash, gemini_pro, claude_sonnet, seed2, kimi
# 模式列表: single (Mode A), uniform10 (Mode B)
#
# 示例:
#   bash run_leaderboard.sh all          # 运行全部 12 组评测 (6模型 × 2模式)
#   bash run_leaderboard.sh single       # 运行全部模型的 single 模式
#   bash run_leaderboard.sh uniform10    # 运行全部模型的 uniform10 模式
#   bash run_leaderboard.sh gemini_flash # 运行 gemini_flash 的两种模式
# ============================================================================

set -e

# ==================== 配置区 ====================
PROJECT_DIR="/root/projects/lql/VG-GUI-Bench"
TEST_JSON="${PROJECT_DIR}/MONDAY/ours_data.json"
DATASET_ROOT="${PROJECT_DIR}/MONDAY"
LOG_ROOT="${PROJECT_DIR}/logs"
TMUX_SESSION="vg_bench"

# 线程数配置（闭源 API 蒸馏平台有严格限频，必须低并发）
THREADS_VLLM=32
THREADS_CLOSED=4

# 所有模型列表
# qwen3vl = Qwen3-VL-235B-A22B-Instruct (本地 vLLM, model_type=qwen3vl in vllm_ips_config.json)
# 闭源模型通过 ClosedModelAdapter 调用 xtools DistillationAPI
ALL_MODELS=("qwen3vl" "gemini_flash" "gemini_pro" "claude_sonnet" "seed2" "kimi")
# 所有模式
ALL_MODES=("single" "uniform10")

# ==================== 环境变量 ====================
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME:${PYTHONPATH:-}"
export PYTHONPATH="$PYTHONPATH:${PROJECT_DIR}/api"

# 清除代理（本地 vLLM 请求不走代理）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# SSM 环境变量（闭源模型 xtools 认证需要）
export SSM_SECURED_ACCESS_KEY_RUNTIME_DEVICE_FINGERPRINT=014d5ac1-b59b024da5ef18-45570e6d54719f-b6637381b29cea-d28f86199bd8ca13
export SSM_SECURED_ACCESS_KEY_PROFILE=/apdcephfs_sh8/share_301266059/aleclv/scripts/profile_1723640549/profile_200004507.json
export SSM_SECURED_ACCESS_KEY_PROFILE_DECRYPT_KEY_DIR=/apdcephfs_sh8/share_301266059/aleclv/scripts/profile_1723640549
export SSM_SECURED_ACCESS_KEY_PROFILE_DECRYPT_KEY_NAME=whitebox.bin

# ==================== 辅助函数 ====================

get_threads() {
    local model=$1
    if [[ "$model" == "qwen3vl" ]]; then
        echo $THREADS_VLLM
    else
        echo $THREADS_CLOSED
    fi
}

get_model_type_arg() {
    # qwen3vl 使用 vllm_tool 的 model_type 配置名
    # 闭源模型直接用适配器名字
    local model=$1
    if [[ "$model" == "qwen3vl" ]]; then
        echo "qwen3vl"
    else
        echo "$model"
    fi
}

run_single_eval() {
    # 运行单个评测任务
    local model=$1
    local mode=$2
    local threads=$(get_threads "$model")
    local model_type=$(get_model_type_arg "$model")
    local eval_name="${model}_${mode}"

    echo "[$(date '+%H:%M:%S')] 启动: model=${model}, mode=${mode}, threads=${threads}"

    cd "$PROJECT_DIR"

    CUDA_VISIBLE_DEVICES=0 python -m core.eval_qwen \
        --use_api \
        --model_type "$model_type" \
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
        --num_threads "$threads"

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
            echo "用法: bash run_leaderboard.sh [all|single|uniform10|模型名]"
            echo "模型名: ${ALL_MODELS[*]}"
            exit 1
        fi
        ;;
esac

echo "============================================================"
echo " VG-GUI-Bench Leaderboard 评测"
echo "============================================================"
echo " 模型: ${RUN_MODELS[*]}"
echo " 模式: ${RUN_MODES[*]}"
echo " 总任务数: $(( ${#RUN_MODELS[@]} * ${#RUN_MODES[@]} ))"
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
            local_threads=$(get_threads "$model")
            local_model_type=$(get_model_type_arg "$model")
            
            # 构建运行命令
            CMD="conda activate guirl_lai_sunqi && "
            CMD+="cd ${PROJECT_DIR} && "
            CMD+="unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && "
            CMD+="export PYTHONUNBUFFERED=1 && "
            CMD+="export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\" && "
            CMD+="export PYTHONPATH=\"\$HOME:\${PYTHONPATH:-}\" && "
            CMD+="export PYTHONPATH=\"\$PYTHONPATH:${PROJECT_DIR}/api\" && "
            CMD+="export SSM_SECURED_ACCESS_KEY_RUNTIME_DEVICE_FINGERPRINT=014d5ac1-b59b024da5ef18-45570e6d54719f-b6637381b29cea-d28f86199bd8ca13 && "
            CMD+="export SSM_SECURED_ACCESS_KEY_PROFILE=/apdcephfs_sh8/share_301266059/aleclv/scripts/profile_1723640549/profile_200004507.json && "
            CMD+="export SSM_SECURED_ACCESS_KEY_PROFILE_DECRYPT_KEY_DIR=/apdcephfs_sh8/share_301266059/aleclv/scripts/profile_1723640549 && "
            CMD+="export SSM_SECURED_ACCESS_KEY_PROFILE_DECRYPT_KEY_NAME=whitebox.bin && "
            CMD+="mkdir -p ${LOG_ROOT}/${local_eval_name}/ && "
            CMD+="CUDA_VISIBLE_DEVICES=0 python -m core.eval_qwen "
            CMD+="--use_api --model_type ${local_model_type} --qwen_name_list all "
            CMD+="--max_tokens 8192 --temperature 0.6 "
            CMD+="--test_json_path ${TEST_JSON} --dataset_root ${DATASET_ROOT} "
            CMD+="--ref_mode ${mode} --log_root ${LOG_ROOT}/${local_eval_name}/ "
            CMD+="--eval_name ${local_eval_name} --task monday --num_history 1 "
            CMD+="--num_threads ${local_threads} "
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
