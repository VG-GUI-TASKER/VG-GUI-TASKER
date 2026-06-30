#!/usr/bin/env python3
"""
整理已选视频到 video_selected/ 目录
- 合并分离的视频流+音频流（ffmpeg）
- 复制字幕文件
- 生成根目录 info.json
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── 配置 ───────────────────────────────────────────────────────────────────────

PIPELINE_DIR   = Path(__file__).parent
VIDEOS_DIR     = PIPELINE_DIR / "videos"
OUTPUT_DIR     = PIPELINE_DIR / "video_selected"
RESULT_JSON    = PIPELINE_DIR / "selection_result.json"

# ffmpeg 路径（剪映内置）
FFMPEG = Path(r"C:\Users\liuqi\AppData\Local\JianyingPro\Apps\9.1.0.13454\ffmpeg.exe")

# ── 工具 ───────────────────────────────────────────────────────────────────────

def find_video_files(task_dir: Path, video_id: str):
    """
    返回 (video_path, audio_path, already_merged)
    - already_merged=True  → video_path 是已合并的完整 mp4，audio_path=None
    - already_merged=False → 需要用 ffmpeg 合并
    """
    files = {f.name: f for f in task_dir.iterdir() if f.is_file()}

    # 1. 已合并的完整 mp4（无格式编号，如 dvbLrwD2SpA.mp4）
    merged = files.get(f"{video_id}.mp4")
    if merged:
        return merged, None, True

    # 2. 视频流：优先 mp4，其次 webm（排除 f251 纯音频）
    video_pat = re.compile(rf"^{re.escape(video_id)}\.(f\d+)\.(mp4|webm)$")
    audio_path = files.get(f"{video_id}.f251.webm")

    video_candidates = []
    for name, path in files.items():
        m = video_pat.match(name)
        if m:
            fmt_code = int(m.group(1)[1:])   # e.g. 399
            ext = m.group(2)
            if fmt_code != 251:               # 排除纯音频流
                video_candidates.append((fmt_code, ext, path))

    if not video_candidates:
        return None, None, False

    # 优先选 mp4，其次 webm；格式编号越大通常质量越好
    video_candidates.sort(key=lambda x: (0 if x[1] == "mp4" else 1, -x[0]))
    _, _, video_path = video_candidates[0]

    return video_path, audio_path, False


def merge_video_audio(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """用 ffmpeg 合并视频流和音频流，输出 mp4。返回是否成功。"""
    if audio_path and audio_path.exists():
        cmd = [
            str(FFMPEG),
            "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(out_path),
        ]
    else:
        # 只有视频流，直接复制容器
        cmd = [
            str(FFMPEG),
            "-y",
            "-i", str(video_path),
            "-c", "copy",
            str(out_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [WARN] ffmpeg 失败: {result.stderr[-300:]}")
        return False
    return True


def find_subtitle(task_dir: Path, video_id: str) -> Path | None:
    for suffix in [f"{video_id}.en.vtt", f"{video_id}.en.srt"]:
        p = task_dir / suffix
        if p.exists():
            return p
    return None


# ── 主逻辑 ─────────────────────────────────────────────────────────────────────

def main():
    if not FFMPEG.exists():
        print(f"❌ 找不到 ffmpeg: {FFMPEG}")
        sys.exit(1)

    # 加载 selection_result.json（含 task / duration）
    with open(RESULT_JSON, encoding="utf-8") as f:
        results = json.load(f)

    # 只保留 status == selected 的条目，按 domain 分组
    by_domain: dict[str, list[dict]] = {}
    for r in results:
        if r["status"] != "selected" or not r.get("selected"):
            continue
        domain = r["domain"]
        by_domain.setdefault(domain, []).append(r)

    # 按 task_id 数字后缀排序，保证 01 02 03 顺序一致
    def task_sort_key(r):
        m = re.search(r"(\d+)$", r["task_id"])
        return int(m.group(1)) if m else 0

    for domain in by_domain:
        by_domain[domain].sort(key=task_sort_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # info.json 内容：{domain: [{id, title, task, duration, path}, ...]}
    info: dict[str, list[dict]] = {}

    total = sum(len(v) for v in by_domain.values())
    done = 0

    for domain, task_list in sorted(by_domain.items()):
        domain_out = OUTPUT_DIR / domain
        info[domain] = []

        for idx, r in enumerate(task_list, start=1):
            done += 1
            task_id   = r["task_id"]
            sel       = r["selected"]
            video_id  = sel["video_id"]
            title     = sel["title"]
            task_desc = r["task"]
            duration  = sel.get("duration", 0)

            # 源目录
            src_dir = VIDEOS_DIR / domain / task_id

            folder_name = f"{idx:02d}"
            dest_dir = domain_out / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{done}/{total}] {domain}/{folder_name}  ({task_id}  {video_id})", end="  ")

            # ── 视频 ──────────────────────────────────────────────────────────
            video_path, audio_path, already_merged = find_video_files(src_dir, video_id)

            dest_video = dest_dir / "video.mp4"

            if video_path is None:
                print("⚠️  找不到视频文件，跳过")
                shutil.rmtree(dest_dir)
                continue

            if already_merged:
                shutil.copy2(video_path, dest_video)
                print("(copy)", end="  ")
            else:
                ok = merge_video_audio(video_path, audio_path, dest_video)
                if not ok:
                    print("❌ 合并失败，跳过")
                    shutil.rmtree(dest_dir)
                    continue
                print("(merge)", end="  ")

            # ── 字幕 ──────────────────────────────────────────────────────────
            sub_src = find_subtitle(src_dir, video_id)
            if sub_src:
                dest_sub = dest_dir / ("video.en.vtt" if sub_src.suffix == ".vtt" else "video.en.srt")
                shutil.copy2(sub_src, dest_sub)
                print("✅")
            else:
                print("(no subtitle) ✅")

            # relative path: after_effects/01/video.mp4
            rel = f"{domain}/{folder_name}/video.mp4"
            info[domain].append({
                "id":       video_id,
                "title":    title,
                "task":     task_desc,
                "duration": duration,
                "path":     rel,
            })

    # ── 写 info.json ─────────────────────────────────────────────────────────
    info_path = OUTPUT_DIR / "info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    total_videos = sum(len(v) for v in info.values())
    print(f"\n✅ 完成！共整理 {total_videos} 个视频")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   info.json: {info_path}")


if __name__ == "__main__":
    main()
