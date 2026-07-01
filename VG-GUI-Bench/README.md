# VG-GUI-Bench

> Part of [**Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction**](../README.md) (ECCV 2026). &nbsp;[← Back to root README](../README.md)

**Video-Grounded GUI Agent Benchmark** — A benchmark for evaluating vision-language models on GUI automation tasks using video tutorial guidance.

## Overview

VG-GUI-Bench evaluates how well multimodal large language models (VLMs) can predict the next GUI action on a mobile device, given reference frames extracted from YouTube tutorial videos and the current screen state. The dataset is released on Hugging Face as [🤗 **Aoraku/VG-GUI-Bench**](https://huggingface.co/datasets/Aoraku/VG-GUI-Bench) and is built on the [MONDAY](https://huggingface.co/datasets/runamu/MONDAY) dataset.

### Key Idea

Given a tutorial video showing how to complete a task on a mobile phone, the model must:
1. Understand the workflow from reference frames (keyframes / uniform samples / annotated frames)
2. Identify the current position in the task from previous action history
3. Predict the **exact next action** (CLICK, SCROLL, TYPE, PRESS, ZOOM, FINISH) on the current screen

## Project Structure

```
VG-GUI-Bench/
├── run.sh                     # Main evaluation entry script
├── run_leaderboard.sh         # Batch run across reference modes for the leaderboard
├── aggregate_leaderboard.py   # Aggregate per-run results into a leaderboard table
├── test_models.py             # Quick connectivity test for your model endpoint
├── api/                       # Model API layer (OpenAI-compatible)
│   ├── model.py               # OpenAICompatibleModel + build_model() factory
│   └── use_api.py             # Usage example / batch calling helper
├── core/                      # Evaluation core
│   ├── eval_qwen.py           # Main evaluation loop (multi-threaded)
│   ├── eval.py                # Scoring engine: action parsing, matching, CSV report
│   ├── prompt.py              # System prompt templates
│   ├── action_matching.py     # Action matching (from Google Android-in-the-Wild)
│   ├── action_type.py         # Action type enums
│   └── models.py              # Model abstraction (OpenAI-compatible)
├── data_process/              # Data preprocessing pipeline
│   ├── process_all.sh         # Generate all image directories
│   ├── extract_scenes.py      # Extract frames by scene timestamps
│   ├── extract_uniform.py     # Uniform frame sampling
│   ├── annotate_monday.py     # Draw red bounding boxes (cropped)
│   ├── annotate_no_cut.py     # Draw red bounding boxes (full-frame)
│   └── utils.py               # ffmpeg utilities, cropping, black border removal
├── annotator/                 # Annotation utilities
├── leaderboard/               # Leaderboard assets
└── MONDAY/                    # Dataset (not included, see Data Preparation)
```

> **Keyframe extraction (the `tasker` mode).** The TASKER algorithm that produces the `tasker` reference frames lives in a separate top-level directory, [`../TASKER/gui/`](../TASKER/README.md). Run it first to generate the frames, then evaluate them here with `bash run.sh tasker cut`.

## Reference Image Modes

The benchmark supports **12 reference modes** to study how different visual context strategies affect agent performance:

| Mode | Description |
|------|-------------|
| `single` | No reference frames; model sees only the current screen |
| `origin` | Scene-timestamp keyframes from video |
| `gt` | Ground-truth annotated frame with red bounding box (oracle) |
| `annotation` | Annotated keyframes as reference |
| `uniform5` / `uniform10` | 5 or 10 uniformly sampled frames |
| `tasker` / `bfs` / `gbfs` / `dijkstra` | Algorithmic keyframe selection strategies |
| `videoagent` / `videotree` | Video understanding agent-based frame selection |

Each mode can run in **cut** (cropped to phone screen) or **nocut** (full frame) variant.

## Supported Actions

| Action | Format | Description |
|--------|--------|-------------|
| CLICK | `CLICK(x, y)` | Tap at normalized coordinates (0.0–1.0) |
| SCROLL | `SCROLL(x1, y1, x2, y2)` | Swipe gesture |
| TYPE | `TYPE("text")` | Text input |
| PRESS | `PRESS("key")` | BACK / HOME / ENTER |
| ZOOM | `ZOOM()` | Multi-touch gesture |
| FINISH | `FINISH()` | Task complete or impossible |

## Data Preparation

1. Download the [VG-GUI-Bench dataset](https://huggingface.co/datasets/Aoraku/VG-GUI-Bench) and place it under `MONDAY/`:

   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="Aoraku/VG-GUI-Bench", repo_type="dataset", local_dir="MONDAY")
   ```

   The released dataset already ships the source videos (`ytb_video/`), the action annotations
   (`ours_data.json`), and 8 pre-rendered image subdirectories under `images/`:
   - `origin` / `origin_no_cut` — Scene-timestamp frames
   - `uniform_5` / `uniform_5_no_cut` — 5 uniformly sampled frames
   - `uniform_10` / `uniform_10_no_cut` — 10 uniformly sampled frames
   - `annotation` / `annotation_no_cut` — Frames with red bounding box annotations

2. (Optional) Regenerate or add reference images from the source videos:
   ```bash
   cd data_process
   bash process_all.sh
   ```

## Usage

### Model Configuration

Evaluation calls any **OpenAI-compatible** endpoint (OpenAI, Azure OpenAI, or a self-hosted server such as vLLM / SGLang / LMDeploy). Configure it via environment variables before running:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-2024-11-20"        # model name to evaluate
# Optional — only for a self-hosted / non-default endpoint:
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

You can verify connectivity first with:

```bash
python test_models.py --model "$OPENAI_MODEL"
```

The `--model_name`, `--api_key`, and `--base_url` flags of `core/eval_qwen.py` (surfaced through `run.sh`) override these environment variables.

### Quick Start

```bash
# Run evaluation with keyframe reference (cropped images)
bash run.sh origin cut

# Run with uniform 10 frames (full frame, 32 threads)
bash run.sh uniform10 nocut 32

# Run without reference images (single mode)
bash run.sh single cut
```

### Available Arguments

```
bash run.sh <mode> <cut|nocut> [num_threads]
```

- `mode`: one of `single`, `origin`, `gt`, `annotation`, `tasker`, `bfs`, `gbfs`, `dijkstra`, `videoagent`, `videotree`, `uniform5`, `uniform10`
- `cut|nocut`: use cropped or full-frame images
- `num_threads`: number of concurrent API threads (default: 16)

### Evaluation Output

Results are saved to `logs/` with the naming pattern `{mode}_{cut/nocut}_{timestamp}`:
- `*_prediction.json` — Raw model predictions
- `*_evaluation.csv` — Per-step evaluation metrics
- `*.log` — Execution logs

## Evaluation Metrics

The scoring system evaluates predictions against ground truth with:
- **Action Type Match** (weight: 30%) — Whether the predicted action type is correct
- **Action Parameter Match** (weight: 70%) — Coordinate accuracy for CLICK/SCROLL, text similarity for TYPE, key match for PRESS
- Support for bounding box IoU matching and distance threshold matching

The action matching implementation is adapted from [Google Android-in-the-Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild).

## Acknowledgements

- [MONDAY Dataset](https://huggingface.co/datasets/runamu/MONDAY)
- [Android-in-the-Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild) for action matching algorithms
