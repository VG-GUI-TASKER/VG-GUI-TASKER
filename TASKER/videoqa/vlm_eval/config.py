"""
Configuration for the VLM-based VideoQA evaluation (EgoSchema & NExT-QA).

This is the strongest / paper-consistent VideoQA setup: TASKER keyframe search
that feeds a multi-image vision-language model (VLM) directly with the selected
frames, together with the Uniform / VideoTree / VideoAgent / Text-only baselines.

Everything is configured through environment variables so that anyone can
reproduce the numbers without editing code:

    # Model endpoint (OpenAI-compatible: OpenAI / Azure / vLLM / SGLang / ...)
    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o-2024-11-20"       # or Qwen3-VL-235B-A22B-Instruct, etc.
    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"   # optional

    # Dataset / output locations (all optional, sensible defaults below)
    export EGOSCHEMA_DIR=/path/to/egoschema        # videos + annotations
    export NEXTQA_DIR=/path/to/nextqa              # videos + test.csv
    export VIDEOQA_RESULTS_DIR=/path/to/results
    export CLIP_MODEL_PATH=openai/clip-vit-large-patch14   # for VideoTree baseline
"""
import os

# ==================== Paths ====================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOQA_DIR = os.path.dirname(_THIS_DIR)  # TASKER/videoqa

# Root that holds the datasets. Defaults to the annotations shipped in videoqa/data.
DATA_ROOT = os.environ.get("VIDEOQA_DATA_ROOT", os.path.join(_VIDEOQA_DIR, "data"))

# EgoSchema: expects videos under EGOSCHEMA_DIR (or a videos/ subdir) and
# subset_anno.json (shipped in videoqa/data/egoschema/subset_anno.json).
EGOSCHEMA_DIR = os.environ.get("EGOSCHEMA_DIR", os.path.join(DATA_ROOT, "egoschema"))

# NExT-QA: expects videos and a test.csv (or MC parquet) under NEXTQA_DIR.
NEXTQA_DIR = os.environ.get("NEXTQA_DIR", os.path.join(DATA_ROOT, "nextqa"))

# Where evaluation results and checkpoints are written.
RESULTS_BASE_DIR = os.environ.get("VIDEOQA_RESULTS_DIR", os.path.join(_THIS_DIR, "results"))

# ==================== Model (VideoTree baseline) ====================
# CLIP model used by the VideoTree baseline. A Hugging Face id downloads on
# first use; point it at a local directory to run fully offline.
CLIP_MODEL_PATH = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")

# ==================== Evaluation hyper-parameters ====================
# Number of frames for the Uniform baseline.
UNIFORM_NUM_FRAMES = 16

# VideoTree parameters
VIDEOTREE_INIT_CLUSTER_NUM = 8
VIDEOTREE_MAX_CLUSTER_NUM = 32
VIDEOTREE_ADAPTIVE_RATE = 2
VIDEOTREE_ITER_THRESHOLD = 4
VIDEOTREE_NUM_SUBCLUSTERS = 4
VIDEOTREE_NUM_SUBSUBCLUSTERS = 4
VIDEOTREE_MAX_FRAMES = 16

# TASKER parameters
TASKER_INIT_INTERVAL = 4          # initial uniform frames (v4 uses frame count)
TASKER_FINAL_STEP = 5             # kept for backward compatibility
TASKER_SEARCH_STRATEGY = "a_star" # a_star / bfs / gbfs / dijkstra
TASKER_MAX_FRAMES = 16            # hard cap on selected frames (TARGET_MAX_FRAMES)

# Number of videos processed in parallel.
MAX_CONCURRENT_REQUESTS = 16

# Sampling FPS when decoding candidate frames.
SAMPLE_FPS = 1.0
