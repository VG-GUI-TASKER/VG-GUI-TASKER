# TASKER for VideoQA — VLM-based evaluation

This directory holds the **VLM-based** VideoQA evaluation used in the ECCV 2026
paper. Instead of relying on pre-extracted text captions, it feeds the frames
selected by **TASKER** *directly* into a multi-image vision-language model (VLM)
and lets the VLM answer the multiple-choice question. It supports both
**EgoSchema** and **NExT-QA**.

> The caption-based pipeline in the parent [`videoqa/`](../) directory is the
> AKeyS-style setup (LLoVi captions + a text LLM, EgoSchema only). This
> `vlm_eval/` directory is a self-contained, model-agnostic evaluation that
> reproduces the Table 1 numbers and covers NExT-QA.

## Methods

| Method | Script | Frame selection |
|--------|--------|-----------------|
| Text-only (blind) | `eval_textonly.py` | none (lower-bound reference) |
| Uniform | `eval_uniform.py` | equally-spaced N frames |
| VideoTree | `eval_videotree.py` | CLIP (ViT-L/14) clustering + VLM relevance scoring |
| VideoAgent | `eval_videoagent.py` | VLM-guided iterative selection (ECCV 2024) |
| **TASKER (ours)** | `eval_tasker.py` | adaptive A\* keyframe search |

All methods share the same final QA prompt (`build_vqa_prompt`) so the
comparison isolates the effect of frame selection.

## Metrics (paper Table 1)

- **EgoSchema**: `Sub.` (subset 500, has ground truth) / `Full` (5031, submit to
  the [official server](https://validation-server.onrender.com/api/upload/)).
- **NExT-QA** (MC test): `Tem.` (Temporal) / `Cau.` (Causal) / `Des.` (Descriptive) / `Avg.`
  Following VideoTree, `TP` is merged into `TN` and `Avg.` is the weighted overall accuracy.

## Setup

Dependencies are shared with the parent `TASKER` project (see
[`../../requirements.txt`](../../requirements.txt)); the VideoTree baseline
additionally needs `torch`, `transformers` and `scipy`.

Configure the model endpoint (any OpenAI-compatible server: OpenAI, Azure,
vLLM, SGLang, LMDeploy, ...) through environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-2024-11-20"          # or Qwen3-VL-235B-A22B-Instruct, ...
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"  # optional, for self-hosted endpoints
```

### Data

| Dataset | What to provide | Where |
|---------|-----------------|-------|
| EgoSchema | videos (`*.mp4`) | `EGOSCHEMA_DIR` (or a `videos/` subdir under it) |
| EgoSchema | `subset_anno.json` | shipped in [`../data/egoschema/`](../data/egoschema/) |
| NExT-QA | videos + `test.csv` (or MC parquet) | `NEXTQA_DIR` |

Point the loaders at your local copies via environment variables:

```bash
export EGOSCHEMA_DIR=/path/to/egoschema
export NEXTQA_DIR=/path/to/nextqa
export VIDEOQA_RESULTS_DIR=/path/to/results      # optional, defaults to ./results
export CLIP_MODEL_PATH=openai/clip-vit-large-patch14   # VideoTree baseline (local dir or HF id)
```

## Run

```bash
# All methods on both datasets (EgoSchema subset + NExT-QA)
bash run_all.sh

# A single method / dataset
bash run_all.sh --method tasker --dataset nextqa
bash run_all.sh --method uniform --dataset egoschema --egoschema_split subset

# EgoSchema full set (predictions for server submission)
bash run_all.sh --method tasker --dataset egoschema --egoschema_split full
```

Or call a script directly:

```bash
python eval_tasker.py --dataset both --max_frames 16 --init_frames 4 --max_workers 16
python eval_tasker.py --dataset nextqa --max_workers 16
```

Every method supports **checkpoint/resume** — results and checkpoints are
written under `VIDEOQA_RESULTS_DIR` (default `./results`), and re-running skips
already-processed samples. Summarise everything with:

```bash
python show_results.py
```

## TASKER hyper-parameters (`eval_tasker.py`)

| Arg | Default | Meaning |
|-----|---------|---------|
| `--max_frames` | 16 | hard cap on selected frames (aligned with Uniform) |
| `--init_frames` | 4 | initial uniform frames (guarantees endpoint coverage) |
| `--search_strategy` | `a_star` | adaptive A\* search |
| `--max_workers` | 16 | number of videos processed in parallel |

The adaptive search prevents A\* from repeatedly zooming into a single
high-motion region: segment selectability is weighted by *question relevance ×
coverage need*, sparse regions are forced when the frame distribution becomes
too imbalanced, and near-duplicate frames are rejected via colour-histogram
similarity.

## Acknowledgments

We thank the authors of [VideoTree](https://github.com/Ziyang412/VideoTree) and
[VideoAgent](https://github.com/wxh1996/VideoAgent); the corresponding baselines
here follow their original algorithms.
