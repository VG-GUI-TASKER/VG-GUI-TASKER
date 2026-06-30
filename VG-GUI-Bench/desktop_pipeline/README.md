# Desktop GUI Video Collection Pipeline

## Overview

本 pipeline 用于从 YouTube 爬取桌面端软件操作教程视频，构建桌面 GUI Agent 的视频数据集。
仿照 [MONDAY](https://arxiv.org/abs/2505.12632) 的 pipeline，但针对桌面端 (Windows/macOS) 软件操作。

## 任务设计统计（v3 精筛版）

### 领域分布（共 250 个任务，目标选出 150 个视频）

| 大类 | 软件 | 任务数 |
|------|------|--------|
| 办公 | Excel(30) + Word(22) + PPT(15) | **67** |
| 设计/创意 | PS(28) + AI(14) + Pr(13) + AE(10) + Blender(15) + Figma(8) | **88** |
| 浏览器 | Chrome(15) + Firefox(5) | **20** |
| CAD/工程 | AutoCAD(18) + Fusion360(11) + SolidWorks(7) | **36** |
| 开发工具 | VS Code(16) + PyCharm(6) + IntelliJ(4) | **26** |
| 操作系统 | Windows(17) + macOS(11) | **28** |
| **总计** | **19款软件** | **250** |

### 难度分布

| 难度 | 数量 | 占比 |
|------|------|------|
| medium | ~115 | 46% |
| **hard** | **~135** | **54%** |

### 涉及的操作类型

- `click` — 左键单击
- `double_click` — 左键双击
- `right_click` — 右键菜单
- `drag` — 拖拽
- `type` — 键盘输入
- `keyboard_shortcut` — 快捷键 (Ctrl+C 等)
- `hover` — 悬停
- `scroll` — 滚动（部分任务隐含）

## 文件说明

```
desktop_pipeline/
├── README.md              # 本文件
├── TASK_LIST.md           # 详细的中文审核清单（250个任务）
├── tasks.json             # 结构化任务数据（待从TASK_LIST确认后更新）
├── download_videos.sh     # YouTube视频批量下载脚本（待生成）
└── filter_metadata.py     # 元数据自动过滤脚本（待生成）
```

## Pipeline 步骤

1. ✅ **任务设计** → `tasks.json`
2. 🔲 **YouTube 视频下载** → `download_videos.sh`
3. 🔲 **元数据自动过滤** → `filter_metadata.py`
4. 🔲 **VLM 内容质量评估**
5. 🔲 **人工终审**
6. 🔲 **自动场景检测**
7. 🔲 **自动动作标注**
8. 🔲 **人工校验**
9. 🔲 **数据格式转换**
