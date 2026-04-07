#!/bin/bash
# ============================================================
# 桌面端 GUI 教程视频批量下载
# ============================================================
#
# 用法:
#   bash download.sh                          # 完整下载（需改下面的 PROXY）
#   bash download.sh --dry-run                # 只打印 query 不下载
#   bash download.sh --domain photoshop       # 只下载 Photoshop 相关
#   bash download.sh --start-from ps_013      # 从某个 task 继续
#
# 注意:
#   1. 需要先安装 yt-dlp: pip install yt-dlp
#   2. 需要设置正确的代理地址
#   3. 下载支持断点续传（Ctrl+C 后重跑会跳过已完成的 task）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# ★ 修改此处的代理地址 ★
# ============================================================
PROXY="socks5://127.0.0.1:1080"

# 默认参数
MAX_PER_TASK=5          # 每个 task 下载几个候选视频
MIN_VIEWS=1000          # 最低观看数
MIN_DURATION=60         # 最短时长(秒) = 1分钟
MAX_DURATION=900        # 最长时长(秒) = 15分钟

python3 "${SCRIPT_DIR}/download_videos.py" \
    --proxy "${PROXY}" \
    --max-per-task ${MAX_PER_TASK} \
    --min-views ${MIN_VIEWS} \
    --min-duration ${MIN_DURATION} \
    --max-duration ${MAX_DURATION} \
    --task-list "${SCRIPT_DIR}/TASK_LIST.md" \
    --output-dir "${SCRIPT_DIR}/videos" \
    "$@"
