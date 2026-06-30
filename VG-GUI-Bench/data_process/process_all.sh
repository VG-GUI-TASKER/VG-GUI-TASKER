#!/bin/bash
# ============================================================
# One-click script to generate all 8 image directories under
# MONDAY/images/ from MONDAY/ytb_video/ videos.
#
# Output directories:
#   origin          - scene timestamp frames, cropped
#   origin_no_cut   - scene timestamp frames, full frame
#   uniform_5       - 5 uniform frames, cropped
#   uniform_10      - 10 uniform frames, cropped
#   uniform_5_no_cut   - 5 uniform frames, full frame
#   uniform_10_no_cut  - 10 uniform frames, full frame
#   annotation         - origin + red bbox annotation
#   annotation_no_cut  - origin_no_cut + red bbox annotation
# ============================================================

set -e

# ---- HuggingFace Mirror ----
export HF_ENDPOINT="https://hf-mirror.com"

# ---- Paths (on the server) ----
PROJECT_ROOT="/data/home/stevefan/projects/lql/VG-GUI-Bench"
MONDAY_ROOT="${PROJECT_ROOT}/MONDAY"
VIDEO_DIR="${MONDAY_ROOT}/ytb_video"
IMAGES_DIR="${MONDAY_ROOT}/images"
JSON_PATH="${MONDAY_ROOT}/ours_data.json"
WORKERS=16

# Ensure we run from the project root so python -m works
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "============================================"
echo "  MONDAY Image Processing Pipeline"
echo "============================================"
echo "VIDEO_DIR:  ${VIDEO_DIR}"
echo "IMAGES_DIR: ${IMAGES_DIR}"
echo "JSON_PATH:  ${JSON_PATH}"
echo "WORKERS:    ${WORKERS}"
echo ""

mkdir -p "${IMAGES_DIR}"

# # ---- 1. origin (scene timestamps, cropped) ----
# echo "========== [1/8] origin (scene timestamps, cropped) =========="
# python -m data_process.extract_scenes \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/origin" \
#     --workers ${WORKERS}

# # ---- 2. origin_no_cut (scene timestamps, full frame) ----
# echo "========== [2/8] origin_no_cut (scene timestamps, full frame) =========="
# python -m data_process.extract_scenes \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/origin_no_cut" \
#     --no_cut \
#     --workers ${WORKERS}

# # ---- 3. uniform_5 (5 uniform frames, cropped) ----
# echo "========== [3/8] uniform_5 (5 uniform frames, cropped) =========="
# python -m data_process.extract_uniform \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/uniform_5" \
#     --num_frames 5 \
#     --workers ${WORKERS}

# # ---- 4. uniform_10 (10 uniform frames, cropped) ----
# echo "========== [4/8] uniform_10 (10 uniform frames, cropped) =========="
# python -m data_process.extract_uniform \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/uniform_10" \
#     --num_frames 10 \
#     --workers ${WORKERS}

# # ---- 5. uniform_5_no_cut (5 uniform frames, full frame) ----
# echo "========== [5/8] uniform_5_no_cut (5 uniform frames, no crop) =========="
# python -m data_process.extract_uniform \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/uniform_5_no_cut" \
#     --num_frames 5 \
#     --no_cut \
#     --workers ${WORKERS}

# # ---- 6. uniform_10_no_cut (10 uniform frames, full frame) ----
# echo "========== [6/8] uniform_10_no_cut (10 uniform frames, no crop) =========="
# python -m data_process.extract_uniform \
#     --video_dir "${VIDEO_DIR}" \
#     --output_dir "${IMAGES_DIR}/uniform_10_no_cut" \
#     --num_frames 10 \
#     --no_cut \
#     --workers ${WORKERS}

# ---- 7. annotation (origin + red bbox, cropped) ----
echo "========== [7/8] annotation (origin + red bbox) =========="
python -m data_process.annotate_monday \
    --json_path "${JSON_PATH}" \
    --images_root "${IMAGES_DIR}/origin" \
    --out_root "${IMAGES_DIR}/annotation"

# ---- 8. annotation_no_cut (origin_no_cut + red bbox, mapped back to full frame) ----
echo "========== [8/8] annotation_no_cut (origin_no_cut + red bbox, remapped) =========="
python -m data_process.annotate_no_cut \
    --json_path "${JSON_PATH}" \
    --images_root "${IMAGES_DIR}/origin_no_cut" \
    --out_root "${IMAGES_DIR}/annotation_no_cut"

echo ""
echo "============================================"
echo "  All done! Summary:"
echo "============================================"
for dir in origin origin_no_cut uniform_5 uniform_10 uniform_5_no_cut uniform_10_no_cut annotation annotation_no_cut; do
    if [ -d "${IMAGES_DIR}/${dir}" ]; then
        count=$(find "${IMAGES_DIR}/${dir}" -name "*.png" -o -name "*.jpg" | wc -l)
        echo "  ${dir}: ${count} images"
    else
        echo "  ${dir}: NOT FOUND"
    fi
done
