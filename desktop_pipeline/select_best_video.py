#!/usr/bin/env python3
"""
VG-GUI-Bench 视频自动选择脚本
================================
从每个任务的候选视频（~5 个）中，根据多维度评分自动选出最佳的 1 个视频。

评分维度（满分 100）：
  1. 标题相关性 (25 分) - 标题与 task description + query 的关键词匹配度
  2. 字幕内容相关性 (30 分) - 字幕文本中包含的任务相关关键词密度
  3. 视频质量 (20 分) - 分辨率、时长是否在最佳范围内、有无完整视频文件
  4. 社区信号 (15 分) - 播放量、点赞数、频道关注数
  5. 文件完整性 (10 分) - 是否有视频文件、字幕文件、info.json

输出：
  - selection_result.json: 每个 task_id 对应的选中视频及其评分
  - review_report.html: 人工审核用的 HTML 报告（低置信度的需要人工复审）

用法：
    python select_best_video.py --videos-dir /path/to/VG-GUI-Bench-Videos
    python select_best_video.py --videos-dir /path/to/VG-GUI-Bench-Videos --output-dir ./results
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============================================================
# 配置
# ============================================================

# 评分权重
WEIGHTS = {
    "title_relevance": 25,
    "subtitle_relevance": 30,
    "video_quality": 20,
    "community_signals": 15,
    "file_completeness": 10,
}

# 理想视频时长范围（秒）—— 120~600 秒的教程视频最好
IDEAL_DURATION_MIN = 120   # 2 分钟
IDEAL_DURATION_MAX = 600   # 10 分钟
# 可接受范围
ACCEPTABLE_DURATION_MIN = 60
ACCEPTABLE_DURATION_MAX = 900

# 置信度阈值：低于此分数的选择需要人工复审
LOW_CONFIDENCE_THRESHOLD = 40
MEDIUM_CONFIDENCE_THRESHOLD = 60

# 高质量分辨率阈值
HIGH_RES_HEIGHT = 1080
MEDIUM_RES_HEIGHT = 720


# ============================================================
# 工具函数
# ============================================================

def load_json(path: Path) -> Optional[dict]:
    """安全加载 JSON 文件，处理各种编码和格式问题"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return json.loads(text)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] 无法解析 {path}: {e}")
        return None


def parse_vtt_text(vtt_path: Path) -> str:
    """
    从 WebVTT 字幕文件中提取纯文本内容。
    去除时间戳、HTML 标签、重复行，返回干净的文本。
    """
    try:
        text = vtt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    lines = text.split("\n")
    clean_lines = []
    seen = set()

    for line in lines:
        # 跳过 WEBVTT 头部
        if line.strip().startswith("WEBVTT") or line.strip().startswith("Kind:") or line.strip().startswith("Language:"):
            continue
        # 跳过时间戳行
        if "-->" in line:
            continue
        # 跳过空行
        stripped = line.strip()
        if not stripped:
            continue

        # 去除 HTML/VTT 标签: <00:00:04.640><c> right</c> -> right
        cleaned = re.sub(r"<[^>]+>", " ", stripped)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            clean_lines.append(cleaned)

    return " ".join(clean_lines).lower()


def tokenize(text: str) -> list:
    """简单分词：小写、去除标点、按空格分割"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def extract_keywords_from_task(task_desc: str, query: str, domain: str) -> set:
    """
    从任务描述和 query 中提取关键词。
    包括：英文 query 中的词 + 中文任务名分词（简单处理）+ 领域相关词
    """
    keywords = set()

    # 从 query 提取（已经是英文）
    query_tokens = tokenize(query)
    # 去掉太常见的停用词
    stopwords = {
        "how", "to", "in", "a", "an", "the", "and", "or", "with", "for",
        "of", "on", "is", "it", "do", "use", "using", "step", "by",
        "tutorial", "guide", "video", "learn", "easy", "simple", "basic",
        "advanced", "beginners", "beginner"
    }
    for token in query_tokens:
        if token not in stopwords:
            keywords.add(token)

    # 从领域名提取
    domain_words = {
        "photoshop": {"photoshop", "ps", "adobe"},
        "office_excel": {"excel", "spreadsheet", "microsoft"},
        "office_word": {"word", "document", "microsoft"},
        "office_powerpoint": {"powerpoint", "ppt", "slides", "presentation", "microsoft"},
        "ide_vscode": {"vscode", "visual", "studio", "code", "editor"},
        "ide_pycharm": {"pycharm", "python", "jetbrains", "ide"},
        "ide_intellij": {"intellij", "idea", "java", "jetbrains", "ide"},
        "blender": {"blender", "3d", "modeling", "render"},
        "after_effects": {"after", "effects", "motion", "graphics", "adobe", "animation"},
        "premiere_pro": {"premiere", "pro", "video", "editing", "adobe"},
        "illustrator": {"illustrator", "vector", "adobe", "design"},
        "figma": {"figma", "design", "prototype", "ui", "ux"},
        "cad_autocad": {"autocad", "cad", "drawing", "drafting", "autodesk"},
        "cad_fusion360": {"fusion", "360", "cad", "3d", "autodesk"},
        "cad_solidworks": {"solidworks", "cad", "3d", "modeling"},
        "os_macos": {"macos", "mac", "apple", "finder", "system"},
        "os_windows": {"windows", "pc", "microsoft", "system", "settings"},
        "web_browser_chrome": {"chrome", "browser", "google", "web"},
        "web_browser_firefox": {"firefox", "browser", "mozilla", "web"},
    }
    if domain in domain_words:
        keywords.update(domain_words[domain])

    return keywords


def compute_keyword_overlap(text_tokens: list, keywords: set) -> float:
    """计算文本 tokens 与关键词集的重叠比例（0~1）"""
    if not keywords:
        return 0.0
    text_set = set(text_tokens)
    overlap = text_set & keywords
    return len(overlap) / len(keywords)


# ============================================================
# 评分函数
# ============================================================

def score_title_relevance(video_info: dict, task_meta: dict, keywords: set) -> float:
    """
    标题相关性评分 (0~1)
    - 标题中包含多少任务关键词
    - 标题是否包含 'tutorial', 'how to' 等教程信号词
    """
    title = video_info.get("title", "")
    title_tokens = tokenize(title)

    # 关键词匹配 (0~0.7)
    overlap = compute_keyword_overlap(title_tokens, keywords)
    keyword_score = min(overlap * 1.4, 0.7)  # 放大一点，因为标题很短

    # 教程信号词加分 (0~0.3)
    tutorial_signals = ["tutorial", "how", "guide", "learn", "step", "beginners", "tips", "tricks"]
    title_lower = title.lower()
    signal_count = sum(1 for s in tutorial_signals if s in title_lower)
    tutorial_score = min(signal_count / 3, 0.3)

    return keyword_score + tutorial_score


def score_subtitle_relevance(subtitle_text: str, keywords: set, task_desc: str) -> float:
    """
    字幕内容相关性评分 (0~1)
    - 字幕中包含多少任务关键词
    - 字幕长度（太短说明内容不丰富）
    - 字幕中操作动词密度（click, select, open, type, drag 等）
    """
    if not subtitle_text:
        return 0.0

    sub_tokens = tokenize(subtitle_text)

    # 关键词匹配 (0~0.5)
    overlap = compute_keyword_overlap(sub_tokens, keywords)
    keyword_score = min(overlap * 1.0, 0.5)

    # 字幕长度分 (0~0.2) - 适中长度最好
    word_count = len(sub_tokens)
    if word_count < 50:
        length_score = 0.05
    elif word_count < 200:
        length_score = 0.1
    elif word_count < 1000:
        length_score = 0.2
    elif word_count < 3000:
        length_score = 0.15
    else:
        length_score = 0.1  # 太长可能不是单一任务的教程

    # 操作动词密度 (0~0.3) - GUI 教程应该有很多操作动词
    action_verbs = {
        "click", "select", "open", "close", "drag", "drop", "type", "press",
        "right-click", "double-click", "scroll", "navigate", "choose", "check",
        "uncheck", "toggle", "enable", "disable", "apply", "save", "export",
        "import", "create", "delete", "add", "remove", "insert", "copy",
        "paste", "cut", "undo", "redo", "zoom", "resize", "move", "adjust",
        "set", "configure", "install", "download", "upload", "tab", "menu",
        "toolbar", "panel", "window", "dialog", "button", "dropdown",
    }
    sub_text_lower = subtitle_text.lower()
    action_count = sum(1 for v in action_verbs if v in sub_text_lower)
    action_score = min(action_count / 15, 0.3)

    return keyword_score + length_score + action_score


def score_video_quality(video_info: dict, info_json: Optional[dict]) -> float:
    """
    视频质量评分 (0~1)
    - 分辨率
    - 时长是否在理想范围
    - 视频编码质量指标
    """
    score = 0.0

    # 分辨率 (0~0.4)
    height = 0
    if info_json:
        height = info_json.get("height", 0) or 0
    if height >= HIGH_RES_HEIGHT:
        score += 0.4
    elif height >= MEDIUM_RES_HEIGHT:
        score += 0.3
    elif height >= 480:
        score += 0.15
    elif height > 0:
        score += 0.05

    # 时长 (0~0.4)
    duration = video_info.get("duration", 0) or 0
    if IDEAL_DURATION_MIN <= duration <= IDEAL_DURATION_MAX:
        score += 0.4
    elif ACCEPTABLE_DURATION_MIN <= duration < IDEAL_DURATION_MIN:
        score += 0.2
    elif IDEAL_DURATION_MAX < duration <= ACCEPTABLE_DURATION_MAX:
        score += 0.25
    elif duration > 0:
        score += 0.05

    # FPS (0~0.1)
    if info_json:
        fps = info_json.get("fps", 0) or 0
        if fps >= 30:
            score += 0.1
        elif fps >= 24:
            score += 0.07
        elif fps > 0:
            score += 0.03

    # 时效性加分 (0~0.1) - 更新的视频可能界面更接近当前版本
    upload_date = video_info.get("upload_date", "")
    if upload_date:
        try:
            year = int(upload_date[:4])
            if year >= 2024:
                score += 0.1
            elif year >= 2022:
                score += 0.07
            elif year >= 2020:
                score += 0.04
        except (ValueError, IndexError):
            pass

    return min(score, 1.0)


def score_community_signals(video_info: dict, info_json: Optional[dict]) -> float:
    """
    社区信号评分 (0~1)
    - 播放量（对数刻度）
    - 点赞数
    - 频道关注数
    """
    score = 0.0

    # 播放量 (0~0.5) - 用对数刻度
    view_count = video_info.get("view_count", 0) or 0
    if view_count > 0:
        log_views = math.log10(max(view_count, 1))
        # 1K -> 3, 10K -> 4, 100K -> 5, 1M -> 6
        if log_views >= 6:  # 1M+
            score += 0.5
        elif log_views >= 5:  # 100K+
            score += 0.4
        elif log_views >= 4:  # 10K+
            score += 0.3
        elif log_views >= 3:  # 1K+
            score += 0.15
        else:
            score += 0.05

    # 点赞数 (0~0.25)
    like_count = 0
    if info_json:
        like_count = info_json.get("like_count", 0) or 0
    if like_count > 0:
        log_likes = math.log10(max(like_count, 1))
        if log_likes >= 4:  # 10K+
            score += 0.25
        elif log_likes >= 3:  # 1K+
            score += 0.2
        elif log_likes >= 2:  # 100+
            score += 0.12
        elif log_likes >= 1:  # 10+
            score += 0.05

    # 频道关注数 (0~0.25)
    followers = 0
    if info_json:
        followers = info_json.get("channel_follower_count", 0) or 0
    if followers > 0:
        log_followers = math.log10(max(followers, 1))
        if log_followers >= 6:  # 1M+
            score += 0.25
        elif log_followers >= 5:  # 100K+
            score += 0.2
        elif log_followers >= 4:  # 10K+
            score += 0.12
        elif log_followers >= 3:  # 1K+
            score += 0.05

    return min(score, 1.0)


def score_file_completeness(task_dir: Path, video_id: str) -> float:
    """
    文件完整性评分 (0~1)
    - 有视频文件 (.mp4/.webm, 排除 .f251.webm 音频流)
    - 有字幕文件 (.en.vtt)
    - 有元数据文件 (.info.json)
    """
    score = 0.0
    files_in_dir = list(task_dir.iterdir()) if task_dir.exists() else []
    file_names = [f.name for f in files_in_dir]

    # 检查视频文件 (0~0.5) - 排除音频流 (.f251.webm)
    has_video = False
    for fn in file_names:
        if fn.startswith(video_id) and (fn.endswith(".mp4") or fn.endswith(".webm")):
            # 排除 f251 (常见的纯音频流)
            if ".f251." not in fn:
                has_video = True
                break
    if has_video:
        score += 0.5

    # 检查字幕文件 (0~0.3)
    has_subtitle = any(
        fn.startswith(video_id) and fn.endswith(".vtt")
        for fn in file_names
    )
    if has_subtitle:
        score += 0.3

    # 检查 info.json (0~0.2)
    has_info = f"{video_id}.info.json" in file_names
    if has_info:
        score += 0.2

    return score


# ============================================================
# 主评分逻辑
# ============================================================

def score_video(video_entry: dict, task_meta: dict, task_dir: Path,
                keywords: set) -> dict:
    """
    对单个视频计算综合评分。
    
    返回 dict:
    {
        "video_id": str,
        "title": str,
        "total_score": float,
        "scores": {dim: score},
        "weighted_scores": {dim: weighted_score},
    }
    """
    video_id = video_entry["video_id"]

    # 加载 info.json
    info_json_path = task_dir / f"{video_id}.info.json"
    info_json = load_json(info_json_path) if info_json_path.exists() else None

    # 加载字幕
    vtt_path = task_dir / f"{video_id}.en.vtt"
    subtitle_text = parse_vtt_text(vtt_path) if vtt_path.exists() else ""

    # 计算各维度分数 (0~1)
    raw_scores = {
        "title_relevance": score_title_relevance(video_entry, task_meta, keywords),
        "subtitle_relevance": score_subtitle_relevance(subtitle_text, keywords, task_meta.get("task", "")),
        "video_quality": score_video_quality(video_entry, info_json),
        "community_signals": score_community_signals(video_entry, info_json),
        "file_completeness": score_file_completeness(task_dir, video_id),
    }

    # 加权计算
    weighted_scores = {}
    total_score = 0.0
    for dim, raw in raw_scores.items():
        w = WEIGHTS[dim]
        weighted = raw * w
        weighted_scores[dim] = round(weighted, 2)
        total_score += weighted

    return {
        "video_id": video_id,
        "title": video_entry.get("title", ""),
        "duration": video_entry.get("duration", 0),
        "view_count": video_entry.get("view_count", 0),
        "channel": video_entry.get("channel", ""),
        "upload_date": video_entry.get("upload_date", ""),
        "total_score": round(total_score, 2),
        "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
        "weighted_scores": weighted_scores,
    }


def process_task(task_dir: Path) -> dict:
    """
    处理一个任务目录：加载 task_meta.json，对每个候选视频评分，选出最佳。
    """
    meta_path = task_dir / "task_meta.json"
    task_meta = load_json(meta_path)
    if not task_meta:
        return {
            "task_id": task_dir.name,
            "status": "error",
            "error": f"无法加载 {meta_path}",
        }

    task_id = task_meta.get("id", task_dir.name)
    domain = task_meta.get("domain", "")
    task_desc = task_meta.get("task", "")
    query = task_meta.get("query", "")
    videos = task_meta.get("videos", [])

    # 无可用视频
    if not videos:
        return {
            "task_id": task_id,
            "domain": domain,
            "task": task_desc,
            "query": query,
            "status": "no_videos",
            "candidates_found": task_meta.get("candidates_found", 0),
            "downloaded": task_meta.get("downloaded", 0),
            "selected": None,
            "all_scores": [],
        }

    # 只有一个视频 -> 直接选中
    keywords = extract_keywords_from_task(task_desc, query, domain)

    all_scored = []
    for video_entry in videos:
        scored = score_video(video_entry, task_meta, task_dir, keywords)
        all_scored.append(scored)

    # 按总分降序排列
    all_scored.sort(key=lambda x: x["total_score"], reverse=True)
    best = all_scored[0]

    # 判断置信度
    if best["total_score"] >= MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "high"
    elif best["total_score"] >= LOW_CONFIDENCE_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    # 如果有多个候选，计算分数差距
    score_gap = None
    if len(all_scored) > 1:
        score_gap = round(all_scored[0]["total_score"] - all_scored[1]["total_score"], 2)

    return {
        "task_id": task_id,
        "domain": domain,
        "task": task_desc,
        "query": query,
        "status": "selected",
        "confidence": confidence,
        "selected": {
            "video_id": best["video_id"],
            "title": best["title"],
            "duration": best["duration"],
            "view_count": best["view_count"],
            "channel": best["channel"],
            "upload_date": best["upload_date"],
            "total_score": best["total_score"],
            "raw_scores": best["raw_scores"],
            "weighted_scores": best["weighted_scores"],
        },
        "score_gap_to_2nd": score_gap,
        "num_candidates": len(all_scored),
        "all_scores": all_scored,
    }


# ============================================================
# 报告生成
# ============================================================

def generate_html_report(results: list, output_path: Path):
    """生成可视化的 HTML 审核报告"""

    # 统计
    total = len(results)
    selected = sum(1 for r in results if r["status"] == "selected")
    no_videos = sum(1 for r in results if r["status"] == "no_videos")
    errors = sum(1 for r in results if r["status"] == "error")
    high_conf = sum(1 for r in results if r.get("confidence") == "high")
    medium_conf = sum(1 for r in results if r.get("confidence") == "medium")
    low_conf = sum(1 for r in results if r.get("confidence") == "low")

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VG-GUI-Bench 视频选择审核报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #1a1a2e; text-align: center; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
.stat-card .number {{ font-size: 2em; font-weight: bold; }}
.stat-card .label {{ color: #666; margin-top: 5px; }}
.stat-card.green .number {{ color: #27ae60; }}
.stat-card.yellow .number {{ color: #f39c12; }}
.stat-card.red .number {{ color: #e74c3c; }}
.stat-card.blue .number {{ color: #3498db; }}

.filters {{ margin: 20px 0; display: flex; gap: 10px; flex-wrap: wrap; }}
.filter-btn {{ padding: 8px 16px; border: 1px solid #ddd; border-radius: 20px; cursor: pointer; background: white; font-size: 14px; }}
.filter-btn.active {{ background: #3498db; color: white; border-color: #3498db; }}

.task-card {{ background: white; margin: 15px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
.task-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
.task-id {{ font-weight: bold; font-size: 1.1em; color: #2c3e50; }}
.task-domain {{ background: #ecf0f1; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
.task-desc {{ color: #555; margin: 5px 0; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
.badge-high {{ background: #d4edda; color: #155724; }}
.badge-medium {{ background: #fff3cd; color: #856404; }}
.badge-low {{ background: #f8d7da; color: #721c24; }}
.badge-no-video {{ background: #e2e3e5; color: #383d41; }}
.badge-error {{ background: #f8d7da; color: #721c24; }}

.score-bar {{ display: flex; align-items: center; margin: 3px 0; }}
.score-label {{ width: 140px; font-size: 0.85em; color: #555; }}
.score-bar-bg {{ flex: 1; background: #ecf0f1; border-radius: 3px; height: 14px; position: relative; }}
.score-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.score-bar-fill.high {{ background: #27ae60; }}
.score-bar-fill.medium {{ background: #f39c12; }}
.score-bar-fill.low {{ background: #e74c3c; }}
.score-value {{ width: 60px; text-align: right; font-size: 0.85em; font-weight: bold; margin-left: 8px; }}

.video-info {{ font-size: 0.9em; color: #666; margin-top: 8px; }}
.video-info a {{ color: #3498db; text-decoration: none; }}
.video-info a:hover {{ text-decoration: underline; }}

.candidates {{ margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px; }}
.candidate {{ display: flex; justify-content: space-between; padding: 5px 0; font-size: 0.85em; border-bottom: 1px solid #f5f5f5; }}
.candidate.selected {{ background: #f0f9ff; font-weight: bold; }}
.candidate .score {{ color: #2c3e50; min-width: 60px; text-align: right; }}
</style>
<script>
function filterBy(status) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.task-card').forEach(card => {{
        if (status === 'all' || card.dataset.status === status || card.dataset.confidence === status) {{
            card.style.display = 'block';
        }} else {{
            card.style.display = 'none';
        }}
    }});
}}
</script>
</head>
<body>
<div class="container">
<h1>🎬 VG-GUI-Bench 视频选择审核报告</h1>
<p style="text-align:center;color:#666;">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="stats">
    <div class="stat-card blue"><div class="number">{total}</div><div class="label">总任务数</div></div>
    <div class="stat-card green"><div class="number">{selected}</div><div class="label">已选择</div></div>
    <div class="stat-card green"><div class="number">{high_conf}</div><div class="label">高置信度</div></div>
    <div class="stat-card yellow"><div class="number">{medium_conf}</div><div class="label">中置信度</div></div>
    <div class="stat-card red"><div class="number">{low_conf}</div><div class="label">低置信度（需复审）</div></div>
    <div class="stat-card red"><div class="number">{no_videos}</div><div class="label">无视频</div></div>
</div>

<div class="filters">
    <button class="filter-btn active" onclick="filterBy('all')">全部 ({total})</button>
    <button class="filter-btn" onclick="filterBy('low')">🔴 低置信度 ({low_conf})</button>
    <button class="filter-btn" onclick="filterBy('medium')">🟡 中置信度 ({medium_conf})</button>
    <button class="filter-btn" onclick="filterBy('high')">🟢 高置信度 ({high_conf})</button>
    <button class="filter-btn" onclick="filterBy('no_videos')">⚪ 无视频 ({no_videos})</button>
</div>
""")

    # 按置信度排序：低 -> 中 -> 高 -> 无视频/错误（方便先审核问题项）
    confidence_order = {"low": 0, "medium": 1, "high": 2}
    status_order = {"error": -1, "no_videos": 3, "selected": 1}

    def sort_key(r):
        if r["status"] != "selected":
            return (status_order.get(r["status"], 99), 0)
        return (confidence_order.get(r.get("confidence", "low"), 99),
                -r["selected"]["total_score"])

    sorted_results = sorted(results, key=sort_key)

    for r in sorted_results:
        task_id = r["task_id"]
        domain = r.get("domain", "")
        task_desc = r.get("task", "")
        status = r["status"]
        confidence = r.get("confidence", "")

        if status == "no_videos":
            badge = '<span class="badge badge-no-video">无视频</span>'
        elif status == "error":
            badge = f'<span class="badge badge-error">错误</span>'
        elif confidence == "high":
            badge = '<span class="badge badge-high">高置信度</span>'
        elif confidence == "medium":
            badge = '<span class="badge badge-medium">中置信度</span>'
        else:
            badge = '<span class="badge badge-low">低置信度</span>'

        html_parts.append(f"""
<div class="task-card" data-status="{status}" data-confidence="{confidence}">
    <div class="task-header">
        <div>
            <span class="task-id">{task_id}</span>
            <span class="task-domain">{domain}</span>
            {badge}
        </div>
    </div>
    <div class="task-desc">任务: {task_desc}</div>
    <div class="task-desc" style="font-size:0.85em;">Query: {r.get("query", "")}</div>
""")

        if status == "selected" and r["selected"]:
            sel = r["selected"]
            total_score = sel["total_score"]
            yt_url = f"https://www.youtube.com/watch?v={sel['video_id']}"

            html_parts.append(f"""
    <div style="margin-top:10px;">
        <strong>选中: </strong>
        <span style="font-size:1.1em;">{sel['title']}</span>
        <span style="font-size:1.3em; font-weight:bold; color: {'#27ae60' if total_score >= 60 else '#f39c12' if total_score >= 40 else '#e74c3c'}; margin-left:10px;">
            {total_score:.1f}/100
        </span>
    </div>
    <div class="video-info">
        📺 <a href="{yt_url}" target="_blank">{sel['video_id']}</a>
        | ⏱️ {sel['duration']}s
        | 👁️ {sel['view_count']:,}
        | 📺 {sel['channel']}
        | 📅 {sel['upload_date']}
    </div>
""")
            # 评分条
            dims_display = {
                "title_relevance": "标题相关性",
                "subtitle_relevance": "字幕内容相关性",
                "video_quality": "视频质量",
                "community_signals": "社区信号",
                "file_completeness": "文件完整性",
            }
            html_parts.append('    <div style="margin-top:10px;">')
            for dim, label in dims_display.items():
                raw = sel["raw_scores"].get(dim, 0)
                weighted = sel["weighted_scores"].get(dim, 0)
                max_w = WEIGHTS[dim]
                pct = (raw * 100)
                bar_class = "high" if pct >= 60 else "medium" if pct >= 30 else "low"
                html_parts.append(f"""
        <div class="score-bar">
            <span class="score-label">{label}</span>
            <div class="score-bar-bg"><div class="score-bar-fill {bar_class}" style="width:{pct:.0f}%"></div></div>
            <span class="score-value">{weighted:.1f}/{max_w}</span>
        </div>""")
            html_parts.append('    </div>')

            # 所有候选列表
            if r.get("all_scores"):
                html_parts.append('    <div class="candidates"><strong>全部候选:</strong>')
                for idx, cand in enumerate(r["all_scores"]):
                    css_class = "candidate selected" if idx == 0 else "candidate"
                    mark = " ✅" if idx == 0 else ""
                    html_parts.append(
                        f'        <div class="{css_class}">'
                        f'<span>{cand["title"][:80]}{"..." if len(cand["title"]) > 80 else ""}{mark}</span>'
                        f'<span class="score">{cand["total_score"]:.1f}</span>'
                        f'</div>'
                    )
                html_parts.append('    </div>')

        elif status == "no_videos":
            html_parts.append(f"""
    <div style="margin-top:10px; color:#e74c3c;">
        ⚠️ 此任务无可用视频 (搜索到 {r.get("candidates_found", "?")} 个候选，下载 {r.get("downloaded", 0)} 个)
    </div>""")

        elif status == "error":
            html_parts.append(f"""
    <div style="margin-top:10px; color:#e74c3c;">
        ❌ 错误: {r.get("error", "未知错误")}
    </div>""")

        html_parts.append("</div>")

    html_parts.append("""
</div>
</body>
</html>""")

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"✅ HTML 审核报告已生成: {output_path}")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VG-GUI-Bench 视频自动选择")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="/data/workspace/fsq_projects/lql/VG-GUI-Bench-Videos",
        help="视频数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与 videos-dir 同级）",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="只处理指定领域（如 photoshop, office_excel）",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"❌ 视频目录不存在: {videos_dir}")
        sys.exit(1)

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VG-GUI-Bench 视频自动选择")
    print("=" * 70)
    print(f"视频目录: {videos_dir}")
    print(f"输出目录: {output_dir}")
    if args.domain:
        print(f"指定领域: {args.domain}")
    print()

    # 加载 tasks.json，构建 id -> task（英文任务描述）映射
    tasks_json_path = Path(__file__).parent / "tasks.json"
    task_name_map: dict = {}
    if tasks_json_path.exists():
        tasks_data = load_json(tasks_json_path)
        if tasks_data and "tasks" in tasks_data:
            for t in tasks_data["tasks"]:
                tid = t.get("id")
                tname = t.get("task")
                if tid and tname:
                    task_name_map[tid] = tname
        print(f"✅ 已加载 tasks.json，共 {len(task_name_map)} 个任务名")
    else:
        print(f"⚠️  未找到 tasks.json: {tasks_json_path}")
    print()

    # 收集所有任务目录
    task_dirs = []
    for domain_dir in sorted(videos_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        if args.domain and domain_dir.name != args.domain:
            continue
        for task_dir in sorted(domain_dir.iterdir()):
            if task_dir.is_dir() and (task_dir / "task_meta.json").exists():
                task_dirs.append(task_dir)

    print(f"共发现 {len(task_dirs)} 个任务")
    print()

    # 逐个处理
    results = []
    for i, task_dir in enumerate(task_dirs, 1):
        task_id = task_dir.name
        domain = task_dir.parent.name
        print(f"[{i}/{len(task_dirs)}] 处理 {domain}/{task_id}...", end=" ")

        result = process_task(task_dir)
        # 用 tasks.json 里的英文任务描述覆盖
        if result.get("task_id") in task_name_map:
            result["task"] = task_name_map[result["task_id"]]
        results.append(result)

        status = result["status"]
        if status == "selected":
            sel = result["selected"]
            conf = result.get("confidence", "?")
            print(f"✅ {sel['total_score']:.1f}分 ({conf}) -> {sel['video_id']}")
        elif status == "no_videos":
            print("⚠️ 无视频")
        else:
            print(f"❌ {result.get('error', '未知错误')}")

    # ---- 汇总统计 ----
    print()
    print("=" * 70)
    print("汇总统计")
    print("=" * 70)

    total = len(results)
    selected = [r for r in results if r["status"] == "selected"]
    no_videos = [r for r in results if r["status"] == "no_videos"]
    errors = [r for r in results if r["status"] == "error"]
    high_conf = [r for r in selected if r.get("confidence") == "high"]
    medium_conf = [r for r in selected if r.get("confidence") == "medium"]
    low_conf = [r for r in selected if r.get("confidence") == "low"]

    print(f"  总任务数:     {total}")
    print(f"  成功选择:     {len(selected)}")
    print(f"    高置信度:   {len(high_conf)}")
    print(f"    中置信度:   {len(medium_conf)}")
    print(f"    低置信度:   {len(low_conf)}")
    print(f"  无视频:       {len(no_videos)}")
    print(f"  错误:         {len(errors)}")

    if selected:
        scores = [r["selected"]["total_score"] for r in selected]
        print(f"\n  分数分布:")
        print(f"    平均:  {sum(scores) / len(scores):.1f}")
        print(f"    中位:  {sorted(scores)[len(scores)//2]:.1f}")
        print(f"    最高:  {max(scores):.1f}")
        print(f"    最低:  {min(scores):.1f}")

    if no_videos:
        print(f"\n  无视频的任务:")
        for r in no_videos:
            print(f"    - {r['task_id']}: {r.get('task', '?')}")

    # ---- 保存结果 ----
    print()

    # 1. 完整结果
    full_result_path = output_dir / "selection_result.json"
    with open(full_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完整结果: {full_result_path}")

    # 2. 精简版（只有 task_id -> video_id 的映射）
    selection_map = {}
    for r in results:
        if r["status"] == "selected" and r["selected"]:
            selection_map[r["task_id"]] = {
                "video_id": r["selected"]["video_id"],
                "title": r["selected"]["title"],
                "score": r["selected"]["total_score"],
                "confidence": r.get("confidence", "unknown"),
                "youtube_url": f"https://www.youtube.com/watch?v={r['selected']['video_id']}",
            }
    map_path = output_dir / "selection_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(selection_map, f, ensure_ascii=False, indent=2)
    print(f"✅ 选择映射: {map_path}  ({len(selection_map)} 个任务)")

    # 3. 需要人工复审的列表
    review_list = []
    for r in results:
        if r["status"] == "no_videos":
            review_list.append({"task_id": r["task_id"], "reason": "no_videos", "task": r.get("task", "")})
        elif r["status"] == "error":
            review_list.append({"task_id": r["task_id"], "reason": "error", "task": r.get("task", "")})
        elif r.get("confidence") == "low":
            review_list.append({
                "task_id": r["task_id"],
                "reason": "low_confidence",
                "task": r.get("task", ""),
                "score": r["selected"]["total_score"],
            })
    if review_list:
        review_path = output_dir / "needs_review.json"
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_list, f, ensure_ascii=False, indent=2)
        print(f"⚠️  需复审: {review_path}  ({len(review_list)} 个任务)")

    # 4. HTML 报告
    html_path = output_dir / "review_report.html"
    generate_html_report(results, html_path)

    print()
    print("🎉 完成！")


if __name__ == "__main__":
    main()
