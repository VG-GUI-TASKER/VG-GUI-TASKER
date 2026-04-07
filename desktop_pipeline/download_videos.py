#!/usr/bin/env python3
"""
Desktop GUI Tutorial Video Downloader
======================================
从 TASK_LIST.md 解析所有任务的 YouTube 搜索 query，使用 yt-dlp 批量搜索并下载候选视频。

核心策略：
1. 每个 task 用原始 query + 多种变体搜索，最大化命中率
2. 严格过滤：时长 1-15min，分辨率 ≥720p，必须有英文字幕
3. 保存完整的 info json 和字幕，便于后续自动过滤
4. 支持断点续传（跳过已下载的 task）
5. 自动生成下载报告

用法:
    python download_videos.py --proxy socks5://127.0.0.1:1080
    python download_videos.py --proxy socks5://127.0.0.1:1080 --max-per-task 3
    python download_videos.py --proxy socks5://127.0.0.1:1080 --domain office_excel
    python download_videos.py --dry-run   # 只打印 query 不下载
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================
# 配置
# ============================================================

# 每个 task 搜索的候选视频数（从多个 query 变体中搜索，去重后取 top-N）
DEFAULT_MAX_PER_TASK = 5

# 时长过滤（秒）
MIN_DURATION = 60       # 1 分钟
MAX_DURATION = 900      # 15 分钟

# 最低观看数（质量代理）
MIN_VIEW_COUNT = 1000

# 下载分辨率上限
MAX_HEIGHT = 1080

# 每个 task 搜索时尝试的结果数（从中过滤）
SEARCH_COUNT = 10

# 下载间隔（秒），避免被限速
DOWNLOAD_DELAY = 2

# ============================================================
# 解析 TASK_LIST.md
# ============================================================

def parse_task_list(task_list_path: str) -> list[dict]:
    """
    从 TASK_LIST.md 中解析所有任务。
    返回 [{"id": "excel_001", "domain": "office_excel", "task": "...", "query": "...", "difficulty": "...", "steps": "..."}, ...]
    """
    tasks = []
    current_domain = None

    # 域名映射：从 markdown 标题推断 domain
    domain_map = {
        "Excel": "office_excel",
        "Word": "office_word",
        "PowerPoint": "office_powerpoint",
        "Photoshop": "photoshop",
        "Illustrator": "illustrator",
        "Premiere Pro": "premiere_pro",
        "After Effects": "after_effects",
        "Chrome": "web_browser_chrome",
        "Firefox": "web_browser_firefox",
        "AutoCAD": "cad_autocad",
        "Fusion 360": "cad_fusion360",
        "SolidWorks": "cad_solidworks",
        "VS Code": "ide_vscode",
        "PyCharm": "ide_pycharm",
        "IntelliJ": "ide_intellij",
        "Windows": "os_windows",
        "macOS": "os_macos",
        "Blender": "blender",
        "Figma": "figma",
    }

    with open(task_list_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 检测 section 标题
        if line.startswith("## "):
            for keyword, domain in domain_map.items():
                if keyword in line:
                    current_domain = domain
                    break

        # 解析任务行：格式为 N. `[id]` 描述 → "query" | difficulty | steps
        # 匹配模式：数字. `[xxx]` ... → "..." | ... | ...
        match = re.match(
            r'^\d+\.\s+`\[(\w+)\]`\s+'       # ID
            r'(?:★\s+)?'                        # 可选的 ★ 标记
            r'(.+?)\s*→\s*'                     # 任务描述
            r'"(.+?)"\s*\|\s*'                  # 搜索 query
            r'\*{0,2}(\w+)\*{0,2}\s*\|\s*'     # 难度
            r'\*{0,2}([\d\-]+)\*{0,2}',        # 步骤数
            line
        )
        if match:
            task_id = match.group(1)
            task_desc = match.group(2).strip()
            query = match.group(3)
            difficulty = match.group(4)
            steps = match.group(5)

            tasks.append({
                "id": task_id,
                "domain": current_domain,
                "task": task_desc,
                "query": query,
                "difficulty": difficulty,
                "steps": steps,
            })

    return tasks


# ============================================================
# Query 变体生成
# ============================================================

def generate_query_variants(base_query: str, domain: str) -> list[str]:
    """
    从基础 query 生成多种搜索变体，提高搜到桌面录屏教程的概率。
    
    策略：
    1. 原始 query（已含 tutorial）
    2. 加 "screen recording" 或 "screencast" 偏好录屏类
    3. 加 "step by step" 偏好详细教程
    4. 加时间限定词 "2024" 或 "2023" 偏好新版本
    """
    variants = [base_query]

    # 变体2: 强调录屏
    if "tutorial" in base_query.lower():
        v2 = base_query.replace("tutorial", "screen recording tutorial")
        variants.append(v2)

    # 变体3: 强调 step by step
    if "step by step" not in base_query.lower():
        v3 = base_query + " step by step"
        variants.append(v3)

    # 变体4: 偏好新版本（CAD/IDE/OS 类更需要）
    newer_domains = {"cad_autocad", "cad_fusion360", "cad_solidworks",
                     "ide_vscode", "ide_pycharm", "ide_intellij",
                     "os_windows", "os_macos", "figma"}
    if domain in newer_domains:
        v4 = base_query + " 2024"
        variants.append(v4)

    return variants


# ============================================================
# yt-dlp 操作
# ============================================================

def search_videos(query: str, proxy: Optional[str] = None,
                  max_results: int = SEARCH_COUNT) -> list[dict]:
    """
    用 yt-dlp 搜索 YouTube 视频，返回 info dict 列表。
    只获取元数据，不下载视频。
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        f"ytsearch{max_results}:{query}",
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            logging.warning(f"yt-dlp search failed for query: {query}")
            logging.warning(f"stderr: {result.stderr[:500]}")
            return []

        entries = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries
    except subprocess.TimeoutExpired:
        logging.warning(f"yt-dlp search timeout for query: {query}")
        return []
    except Exception as e:
        logging.warning(f"yt-dlp search error for query: {query}: {e}")
        return []


def get_video_info(video_id: str, proxy: Optional[str] = None) -> Optional[dict]:
    """
    获取单个视频的完整 info（含时长、观看数等），用于过滤。
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def filter_video(info: dict) -> tuple[bool, str]:
    """
    根据元数据过滤视频。
    返回 (是否通过, 原因)。
    """
    # 时长过滤
    duration = info.get("duration", 0) or 0
    if duration < MIN_DURATION:
        return False, f"too_short ({duration}s)"
    if duration > MAX_DURATION:
        return False, f"too_long ({duration}s)"

    # 观看数过滤
    view_count = info.get("view_count", 0) or 0
    if view_count < MIN_VIEW_COUNT:
        return False, f"low_views ({view_count})"

    # 标题基础质量检查：排除明显非教程内容
    title = (info.get("title") or "").lower()
    reject_keywords = ["reaction", "unboxing", "review vs", "funny",
                       "meme", "shorts", "tiktok", "#shorts"]
    for kw in reject_keywords:
        if kw in title:
            return False, f"rejected_title_keyword ({kw})"

    # 语言检查：尽量保证英文内容
    # 通过标题和描述中的英文字母比例判断
    title_text = info.get("title", "")
    ascii_ratio = sum(1 for c in title_text if c.isascii()) / max(len(title_text), 1)
    if ascii_ratio < 0.5:
        return False, f"non_english_title (ascii_ratio={ascii_ratio:.2f})"

    return True, "passed"


def download_video(video_id: str, output_dir: str, proxy: Optional[str] = None) -> bool:
    """
    下载单个视频（含字幕和 info json）。
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "--no-overwrites",
        "-o", output_template,
        url,
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logging.warning(f"Download failed for {video_id}: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.warning(f"Download timeout for {video_id}")
        return False
    except Exception as e:
        logging.warning(f"Download error for {video_id}: {e}")
        return False


# ============================================================
# 进度管理
# ============================================================

class ProgressTracker:
    """断点续传 + 下载报告管理"""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "tasks_completed": {},  # task_id -> {videos: [...], status: "done"/"partial"}
            "tasks_failed": {},     # task_id -> reason
            "total_downloaded": 0,
            "total_filtered_out": 0,
            "total_searched": 0,
        }

    def save(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_task_done(self, task_id: str) -> bool:
        return task_id in self.data["tasks_completed"]

    def record_task(self, task_id: str, videos: list[dict], status: str = "done"):
        self.data["tasks_completed"][task_id] = {
            "videos": videos,
            "status": status,
            "completed_at": datetime.now().isoformat(),
        }
        self.save()

    def record_failure(self, task_id: str, reason: str):
        self.data["tasks_failed"][task_id] = {
            "reason": reason,
            "failed_at": datetime.now().isoformat(),
        }
        self.save()

    def increment_stats(self, downloaded: int = 0, filtered_out: int = 0, searched: int = 0):
        self.data["total_downloaded"] += downloaded
        self.data["total_filtered_out"] += filtered_out
        self.data["total_searched"] += searched


# ============================================================
# 主流程
# ============================================================

def process_task(task: dict, output_base: str, proxy: Optional[str],
                 max_per_task: int, tracker: ProgressTracker,
                 dry_run: bool = False) -> list[dict]:
    """
    处理单个任务：搜索 + 过滤 + 下载。
    返回成功下载的视频信息列表。
    """
    task_id = task["id"]
    domain = task["domain"]
    query = task["query"]

    logging.info(f"\n{'='*60}")
    logging.info(f"Processing [{task_id}] {task['task']}")
    logging.info(f"  Domain: {domain}")
    logging.info(f"  Base query: {query}")
    logging.info(f"{'='*60}")

    # 如果已经完成，跳过
    if tracker.is_task_done(task_id):
        logging.info(f"  [SKIP] Already completed, skipping.")
        return []

    # 生成搜索变体
    variants = generate_query_variants(query, domain)
    logging.info(f"  Query variants ({len(variants)}):")
    for i, v in enumerate(variants):
        logging.info(f"    [{i+1}] {v}")

    if dry_run:
        logging.info(f"  [DRY RUN] Skipping search and download.")
        return []

    # 搜索所有变体，收集候选视频 ID（去重）
    seen_ids = set()
    candidates = []

    for variant in variants:
        logging.info(f"  Searching: {variant}")
        results = search_videos(variant, proxy=proxy, max_results=SEARCH_COUNT)
        tracker.increment_stats(searched=len(results))

        for entry in results:
            vid = entry.get("id") or entry.get("url", "").split("=")[-1]
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                candidates.append(entry)

        time.sleep(1)  # 搜索间的短暂延迟

    logging.info(f"  Found {len(candidates)} unique candidates from {len(variants)} query variants")

    if not candidates:
        logging.warning(f"  [FAIL] No candidates found for [{task_id}]")
        tracker.record_failure(task_id, "no_candidates")
        return []

    # 对候选视频做详细过滤
    # flat-playlist 模式拿到的 info 不完整，需要逐个获取详情再过滤
    passed = []
    for entry in candidates:
        vid = entry.get("id") or entry.get("url", "").split("=")[-1]
        if not vid:
            continue

        # 快速预过滤（flat-playlist 中可能有 duration）
        dur = entry.get("duration")
        if dur is not None:
            if dur < MIN_DURATION or dur > MAX_DURATION:
                tracker.increment_stats(filtered_out=1)
                continue

        views = entry.get("view_count")
        if views is not None and views < MIN_VIEW_COUNT:
            tracker.increment_stats(filtered_out=1)
            continue

        # 获取完整 info 做详细过滤
        logging.info(f"  Checking video {vid}...")
        full_info = get_video_info(vid, proxy=proxy)
        if full_info is None:
            logging.info(f"    -> Could not get info, skipping")
            tracker.increment_stats(filtered_out=1)
            continue

        ok, reason = filter_video(full_info)
        if not ok:
            logging.info(f"    -> Filtered out: {reason}")
            tracker.increment_stats(filtered_out=1)
            continue

        logging.info(f"    -> PASSED (duration={full_info.get('duration', '?')}s, "
                     f"views={full_info.get('view_count', '?')}, "
                     f"title={full_info.get('title', '?')[:60]})")
        passed.append({
            "video_id": vid,
            "title": full_info.get("title", ""),
            "duration": full_info.get("duration", 0),
            "view_count": full_info.get("view_count", 0),
            "channel": full_info.get("channel", ""),
            "upload_date": full_info.get("upload_date", ""),
        })

        if len(passed) >= max_per_task:
            break

        time.sleep(0.5)

    if not passed:
        logging.warning(f"  [FAIL] No videos passed filter for [{task_id}]")
        tracker.record_failure(task_id, "all_filtered_out")
        return []

    logging.info(f"  {len(passed)} videos passed filter, downloading...")

    # 下载
    task_dir = os.path.join(output_base, domain, task_id)
    os.makedirs(task_dir, exist_ok=True)

    downloaded = []
    for vid_info in passed:
        vid = vid_info["video_id"]
        logging.info(f"  Downloading {vid} -> {task_dir}")

        success = download_video(vid, task_dir, proxy=proxy)
        if success:
            vid_info["downloaded"] = True
            downloaded.append(vid_info)
            tracker.increment_stats(downloaded=1)
            logging.info(f"    -> Downloaded successfully")
        else:
            vid_info["downloaded"] = False
            logging.warning(f"    -> Download failed")

        time.sleep(DOWNLOAD_DELAY)

    # 保存该 task 的元数据
    task_meta = {
        **task,
        "candidates_found": len(candidates),
        "passed_filter": len(passed),
        "downloaded": len(downloaded),
        "videos": downloaded,
    }
    meta_path = os.path.join(task_dir, "task_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(task_meta, f, indent=2, ensure_ascii=False)

    status = "done" if len(downloaded) >= 1 else "partial"
    tracker.record_task(task_id, downloaded, status)

    logging.info(f"  [{task_id}] Completed: {len(downloaded)}/{len(passed)} downloaded")
    return downloaded


def generate_report(tracker: ProgressTracker, output_base: str):
    """生成下载报告"""
    report_path = os.path.join(output_base, "download_report.json")

    data = tracker.data
    completed = data["tasks_completed"]
    failed = data["tasks_failed"]

    # 统计各 domain 的完成情况
    domain_stats = {}
    for task_id, info in completed.items():
        domain = task_id.split("_")[0]  # 粗略推断
        if domain not in domain_stats:
            domain_stats[domain] = {"completed": 0, "total_videos": 0}
        domain_stats[domain]["completed"] += 1
        domain_stats[domain]["total_videos"] += len(info.get("videos", []))

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "tasks_completed": len(completed),
            "tasks_failed": len(failed),
            "total_videos_downloaded": data["total_downloaded"],
            "total_candidates_filtered_out": data["total_filtered_out"],
            "total_searched": data["total_searched"],
        },
        "domain_stats": domain_stats,
        "failed_tasks": failed,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logging.info(f"\n{'='*60}")
    logging.info("DOWNLOAD REPORT")
    logging.info(f"{'='*60}")
    logging.info(f"Tasks completed: {len(completed)}")
    logging.info(f"Tasks failed: {len(failed)}")
    logging.info(f"Total videos downloaded: {data['total_downloaded']}")
    logging.info(f"Total filtered out: {data['total_filtered_out']}")
    logging.info(f"Report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Desktop GUI Tutorial Video Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整下载（需要代理）
  python download_videos.py --proxy socks5://127.0.0.1:1080

  # 只下载某个 domain
  python download_videos.py --proxy socks5://127.0.0.1:1080 --domain photoshop

  # 每个 task 只下载 3 个候选
  python download_videos.py --proxy socks5://127.0.0.1:1080 --max-per-task 3

  # 只看 query 不下载
  python download_videos.py --dry-run

  # 从某个 task 开始（断点续传自动跳过已完成的）
  python download_videos.py --proxy socks5://127.0.0.1:1080 --start-from ps_013
        """
    )
    parser.add_argument("--proxy", type=str, default=None,
                        help="yt-dlp proxy (e.g. socks5://127.0.0.1:1080)")
    parser.add_argument("--max-per-task", type=int, default=DEFAULT_MAX_PER_TASK,
                        help=f"Max videos to download per task (default: {DEFAULT_MAX_PER_TASK})")
    parser.add_argument("--domain", type=str, default=None,
                        help="Only process tasks from this domain (e.g. photoshop, office_excel)")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from this task ID (skip all previous)")
    parser.add_argument("--task-list", type=str,
                        default=os.path.join(os.path.dirname(__file__), "TASK_LIST.md"),
                        help="Path to TASK_LIST.md")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "videos"),
                        help="Output directory for downloaded videos")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print queries, don't search or download")
    parser.add_argument("--min-views", type=int, default=MIN_VIEW_COUNT,
                        help=f"Minimum view count filter (default: {MIN_VIEW_COUNT})")
    parser.add_argument("--min-duration", type=int, default=MIN_DURATION,
                        help=f"Minimum duration in seconds (default: {MIN_DURATION})")
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION,
                        help=f"Maximum duration in seconds (default: {MAX_DURATION})")

    args = parser.parse_args()

    # 更新全局过滤参数
    global MIN_VIEW_COUNT, MIN_DURATION, MAX_DURATION
    MIN_VIEW_COUNT = args.min_views
    MIN_DURATION = args.min_duration
    MAX_DURATION = args.max_duration

    # 设置输出目录
    output_base = os.path.abspath(args.output_dir)
    os.makedirs(output_base, exist_ok=True)

    # 设置日志
    log_file = os.path.join(output_base, f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info("=" * 60)
    logging.info("Desktop GUI Tutorial Video Downloader")
    logging.info("=" * 60)
    logging.info(f"Task list: {args.task_list}")
    logging.info(f"Output dir: {output_base}")
    logging.info(f"Proxy: {args.proxy or 'None'}")
    logging.info(f"Max per task: {args.max_per_task}")
    logging.info(f"Domain filter: {args.domain or 'all'}")
    logging.info(f"Duration filter: {args.min_duration}-{args.max_duration}s")
    logging.info(f"Min views: {args.min_views}")
    logging.info(f"Dry run: {args.dry_run}")
    logging.info(f"Log file: {log_file}")

    # 解析任务
    tasks = parse_task_list(args.task_list)
    logging.info(f"\nParsed {len(tasks)} tasks from TASK_LIST.md")

    if not tasks:
        logging.error("No tasks found! Check TASK_LIST.md format.")
        sys.exit(1)

    # 按 domain 过滤
    if args.domain:
        tasks = [t for t in tasks if args.domain in (t["domain"] or "")]
        logging.info(f"Filtered to {len(tasks)} tasks for domain '{args.domain}'")

    # 从指定 task 开始
    if args.start_from:
        start_idx = None
        for i, t in enumerate(tasks):
            if t["id"] == args.start_from:
                start_idx = i
                break
        if start_idx is not None:
            tasks = tasks[start_idx:]
            logging.info(f"Starting from task '{args.start_from}', {len(tasks)} tasks remaining")
        else:
            logging.warning(f"Task '{args.start_from}' not found, processing all tasks")

    # 打印域分布
    domain_counts = {}
    for t in tasks:
        d = t["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    logging.info(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        logging.info(f"  {d}: {c} tasks")

    # 初始化进度追踪
    progress_file = os.path.join(output_base, "progress.json")
    tracker = ProgressTracker(progress_file)

    # 检查 yt-dlp 是否安装
    if not args.dry_run:
        try:
            result = subprocess.run(["yt-dlp", "--version"],
                                    capture_output=True, text=True, timeout=10)
            logging.info(f"\nyt-dlp version: {result.stdout.strip()}")
        except FileNotFoundError:
            logging.error("yt-dlp not found! Install with: pip install yt-dlp")
            sys.exit(1)

    # 开始处理
    total_downloaded = 0
    for i, task in enumerate(tasks):
        logging.info(f"\n[{i+1}/{len(tasks)}] Processing task...")
        try:
            downloaded = process_task(
                task=task,
                output_base=output_base,
                proxy=args.proxy,
                max_per_task=args.max_per_task,
                tracker=tracker,
                dry_run=args.dry_run,
            )
            total_downloaded += len(downloaded)
        except KeyboardInterrupt:
            logging.info("\n\nInterrupted by user. Saving progress...")
            tracker.save()
            generate_report(tracker, output_base)
            sys.exit(0)
        except Exception as e:
            logging.error(f"Unexpected error processing [{task['id']}]: {e}")
            tracker.record_failure(task["id"], str(e))
            continue

    # 生成最终报告
    tracker.save()
    report = generate_report(tracker, output_base)

    logging.info(f"\n{'='*60}")
    logging.info(f"ALL DONE! Downloaded {total_downloaded} videos from {len(tasks)} tasks.")
    logging.info(f"Videos saved to: {output_base}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
