#!/usr/bin/env python3
"""
把 results.txt 中的扩写任务描述回填到 video_selected/info.json
"""

import json
import re
from pathlib import Path

RESULTS_TXT = Path(__file__).parent / "results.txt"
INFO_JSON   = Path(__file__).parent / "video_selected" / "info.json"

# 解析 results.txt：[category/nn] task text
entries = {}
pattern = re.compile(r"^\[([a-z_0-9]+/\d+)\]\s+(.+)$")

for line in RESULTS_TXT.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    m = pattern.match(line)
    if m:
        entries[m.group(1)] = m.group(2).strip()

print(f"解析到 {len(entries)} 条任务描述")

# 回填 info.json
with open(INFO_JSON, encoding="utf-8") as f:
    info = json.load(f)

updated = 0
missing = []

for domain, videos in info.items():
    for i, v in enumerate(videos, 1):
        key = f"{domain}/{i:02d}"
        if key in entries:
            v["task"] = entries[key]
            updated += 1
        else:
            missing.append(key)

if missing:
    print(f"⚠️  未找到对应条目: {missing}")

with open(INFO_JSON, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"✅ 已更新 {updated} 条任务描述 → {INFO_JSON}")
