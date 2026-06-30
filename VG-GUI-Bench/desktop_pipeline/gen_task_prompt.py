#!/usr/bin/env python3
"""
从 info.json 提取所有任务，生成两个文件：
  1. tasks_raw.txt       — 原始任务清单（供参考）
  2. expand_tasks_prompt.txt — 发给 Claude chatbox 的 prompt
"""

import json
from pathlib import Path

INFO_JSON = Path(__file__).parent / "video_selected" / "info.json"
OUT_DIR   = Path(__file__).parent

SOFTWARE_NAMES = {
    "after_effects":      "Adobe After Effects",
    "blender":            "Blender",
    "cad_autocad":        "AutoCAD",
    "cad_fusion360":      "Autodesk Fusion 360",
    "cad_solidworks":     "SolidWorks",
    "figma":              "Figma",
    "ide_intellij":       "IntelliJ IDEA",
    "ide_pycharm":        "PyCharm",
    "ide_vscode":         "Visual Studio Code",
    "illustrator":        "Adobe Illustrator",
    "office_excel":       "Microsoft Excel",
    "office_powerpoint":  "Microsoft PowerPoint",
    "office_word":        "Microsoft Word",
    "os_macos":           "macOS",
    "os_windows":         "Windows 11",
    "photoshop":          "Adobe Photoshop",
    "premiere_pro":       "Adobe Premiere Pro",
    "web_browser_chrome": "Google Chrome",
    "web_browser_firefox":"Mozilla Firefox",
}

with open(INFO_JSON, encoding="utf-8") as f:
    info = json.load(f)

# ── 1. tasks_raw.txt ──────────────────────────────────────────────────────────
raw_lines = []
all_entries = []   # (category, idx, software, task, video_title)

for domain, videos in info.items():
    software = SOFTWARE_NAMES.get(domain, domain)
    raw_lines.append(f"\n## {domain}  [{software}]")
    for i, v in enumerate(videos, 1):
        line = f"  {i:02d}. {v['task']}"
        raw_lines.append(line)
        all_entries.append({
            "key":      f"{domain}/{i:02d}",
            "domain":   domain,
            "software": software,
            "task":     v["task"],
            "title":    v["title"],
        })

(OUT_DIR / "tasks_raw.txt").write_text(
    "\n".join(raw_lines).strip(), encoding="utf-8"
)
print(f"✅ tasks_raw.txt  ({len(all_entries)} tasks)")

# ── 2. expand_tasks_prompt.txt ────────────────────────────────────────────────
task_block = []
for e in all_entries:
    task_block.append(f"[{e['key']}] ({e['software']}) {e['task']}")

task_text = "\n".join(task_block)

prompt = f"""# Task: Rewrite GUI benchmark task descriptions

## Background

I am building **VG-GUI-Bench**, a video-guided GUI benchmark designed to evaluate Vision-Language Models (VLMs) on their ability to complete long-horizon desktop software tasks by following instructional tutorial videos. Each benchmark item consists of:
- A tutorial video demonstrating a software workflow
- A task description that will be given to the VLM as its goal
- A sequence of GUI actions the model must perform to accomplish the task

The task descriptions are currently written as brief phrases (some in Chinese). I need each one rewritten as a **clear, natural English question** in the form:

> How to use [software name] to [specific action/goal]?

## Requirements

1. **Language**: All output must be in English.
2. **Format**: Each task must be a complete question: `How to use [software] to [action]?`
3. **Specificity**: Be specific enough that a user could search for this exact tutorial on YouTube and find the right video. Avoid generic phrasing.
4. **Scope**: The task should describe a single, focused workflow — not a full course.
5. **Tone**: Neutral, instructional.
6. **Key prefix**: Keep the `[category/number]` prefix exactly as-is at the start of each line so I can parse the output back.

## Output format

Return ONLY the rewritten list, one entry per line, like this:
```
[after_effects/01] How to use Adobe After Effects to create a motion graphics title animation from scratch?
[after_effects/02] ...
```

Do NOT include any explanation, commentary, or markdown outside the code block.

## Task list ({len(all_entries)} items)

```
{task_text}
```
"""

(OUT_DIR / "expand_tasks_prompt.txt").write_text(prompt, encoding="utf-8")
print(f"✅ expand_tasks_prompt.txt  ({len(all_entries)} entries in prompt)")
