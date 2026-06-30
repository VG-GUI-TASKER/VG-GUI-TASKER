"""
Draw red bounding boxes on UNCROPPED (no_cut) MONDAY frames.

The annot_position in ours_data.json is relative to the CROPPED image
(after screen_bbox crop + black-border removal). This script reverses
that mapping to place the boxes correctly on the full original frame.

Iterates over ALL images found in --images_root (origin_no_cut/), ensuring
output count == input count.
"""
import os
import json
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from datasets import load_dataset

from data_process.utils import get_pure_size


def build_bbox_map(video_ids=None):
    """Build video_id -> list of screen_bboxes from HF dataset.

    Args:
        video_ids: optional set of video_ids to filter. If None, all videos are included.
    """
    print("Loading MONDAY from HuggingFace Hub for screen_bboxes...")
    bbox_map = {}
    dataset_dict = load_dataset("runamu/MONDAY")
    for split_name in dataset_dict.keys():
        for row in dataset_dict[split_name]:
            vid = row["video_id"]
            if video_ids is not None and vid not in video_ids:
                continue
            if vid not in bbox_map:
                bbox_map[vid] = row["screen_bboxes"]
    if video_ids is not None:
        print(f"  Got screen_bboxes for {len(bbox_map)}/{len(video_ids)} local videos.")
    return bbox_map


def build_annot_index(json_path: str) -> Dict[str, list]:
    """Build a mapping: img_filename -> annot_position list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("ours", [])
    index = {}
    for ep in episodes:
        if not isinstance(ep, list):
            continue
        for step in ep:
            img_filename = step.get("img_filename", "")
            if not img_filename:
                continue
            action_list = step.get("action_list", []) or []
            if not isinstance(action_list, list):
                action_list = [action_list]
            annot_position = []
            if len(action_list) > 0:
                action = action_list[0]
                if isinstance(action, dict):
                    annot_position = action.get("annot_position", []) or []
            index[img_filename] = annot_position
    return index


def scan_all_images(images_root: str) -> List[Tuple[str, str, str]]:
    """Scan images_root for all image files.

    Returns list of (video_id, frame_base_no_ext, full_path).
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    results = []
    for video_id in sorted(os.listdir(images_root)):
        video_dir = os.path.join(images_root, video_id)
        if not os.path.isdir(video_dir):
            continue
        for fname in sorted(os.listdir(video_dir)):
            base, ext = os.path.splitext(fname)
            if ext.lower() in exts:
                results.append((video_id, base, os.path.join(video_dir, fname)))
    return results


def compute_pure_region_in_full_frame(full_img: Image.Image, screen_bbox: list):
    """
    Given the full frame and a screen_bbox (left, top, right, bottom),
    compute the absolute pixel coordinates of the pure-content region
    (after screen crop + black-border removal) within the full frame.

    Returns (pure_left, pure_top, pure_right, pure_bottom) in full-frame pixels,
    and (pure_W, pure_H) = size of the pure region.
    """
    sb_left, sb_top, sb_right, sb_bottom = screen_bbox
    cropped = full_img.crop((sb_left, sb_top, sb_right, sb_bottom))

    frame_np = np.array(cropped)
    cH, cW = frame_np.shape[:2]
    p_x, p_y, p_w, p_h = get_pure_size(frame_np)

    pure_cx1 = int(p_x * cW)
    pure_cy1 = int(p_y * cH)
    pure_cx2 = int((p_x + p_w) * cW)
    pure_cy2 = int((p_y + p_h) * cH)

    pure_left = sb_left + pure_cx1
    pure_top = sb_top + pure_cy1
    pure_right = sb_left + pure_cx2
    pure_bottom = sb_top + pure_cy2

    pure_W = pure_right - pure_left
    pure_H = pure_bottom - pure_top

    return (pure_left, pure_top, pure_right, pure_bottom), (pure_W, pure_H)


def annot_to_fullframe_boxes(
    annot_position: List[float],
    pure_region: Tuple[int, int, int, int],
    pure_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """
    Map annot_position (normalized to the pure cropped image) to
    absolute pixel boxes in the full frame.
    """
    if not annot_position:
        return []

    pr_left, pr_top, pr_right, pr_bottom = pure_region
    pW, pH = pure_size

    usable_len = (len(annot_position) // 4) * 4
    annot_position = annot_position[:usable_len]

    boxes = []
    for i in range(0, usable_len, 4):
        y, x, h, w = annot_position[i:i + 4]
        y = max(0.0, min(1.0, y))
        x = max(0.0, min(1.0, x))
        h = max(0.0, min(1.0, h))
        w = max(0.0, min(1.0, w))

        box_left_in_pure = int(math.floor(x * pW))
        box_top_in_pure = int(math.floor(y * pH))
        box_right_in_pure = int(math.ceil((x + w) * pW))
        box_bottom_in_pure = int(math.ceil((y + h) * pH))

        full_left = pr_left + box_left_in_pure
        full_top = pr_top + box_top_in_pure
        full_right = pr_left + box_right_in_pure
        full_bottom = pr_top + box_bottom_in_pure

        full_left = max(0, full_left)
        full_top = max(0, full_top)
        full_right = max(full_left + 1, full_right)
        full_bottom = max(full_top + 1, full_bottom)

        boxes.append((full_left, full_top, full_right, full_bottom))

    return boxes


def draw_boxes_on_image(img: Image.Image, boxes: List[Tuple[int, int, int, int]], width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (l, t, r, b) in boxes:
        draw.rectangle([l, t, r, b], outline="red", width=width)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw red bboxes on uncropped MONDAY frames (mapping annot coords back to full frame)"
    )
    parser.add_argument("--json_path", type=str, required=True, help="Path to ours_data.json")
    parser.add_argument("--images_root", type=str, required=True, help="Root dir of uncropped images (origin_no_cut/)")
    parser.add_argument("--out_root", type=str, required=True, help="Output dir for annotated images")
    parser.add_argument("--line_width", type=int, default=3, help="Rectangle line width")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Build annotation index from JSON
    annot_index = build_annot_index(args.json_path)
    print(f"Loaded {len(annot_index)} annotation entries from JSON")

    # Scan all images in images_root
    all_images = scan_all_images(args.images_root)
    print(f"Found {len(all_images)} images in {args.images_root}")

    # Collect video_ids that appear in images_root
    local_video_ids = set(vid for vid, _, _ in all_images)

    # Load screen_bboxes from HF dataset
    bbox_map = build_bbox_map(video_ids=local_video_ids)

    saved = 0
    annotated_count = 0
    no_bbox = 0

    for video_id, frame_base, img_path in tqdm(all_images, desc="Annotating (no_cut)"):
        output_video_dir = os.path.join(args.out_root, video_id)
        os.makedirs(output_video_dir, exist_ok=True)

        out_name = f"{frame_base}_annot.png"
        out_path = os.path.join(output_video_dir, out_name)

        if (not args.overwrite) and os.path.exists(out_path):
            saved += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Look up annotation
        lookup_key = f"{video_id}/{frame_base}"
        annot_position = annot_index.get(lookup_key, [])

        # Get screen_bbox for this frame
        frame_idx = None
        try:
            frame_idx = int(frame_base.split("_")[-1])
        except (ValueError, IndexError):
            pass

        video_bboxes = bbox_map.get(video_id, [])
        if frame_idx is not None and frame_idx < len(video_bboxes):
            screen_bbox = video_bboxes[frame_idx]
        elif len(video_bboxes) > 0:
            screen_bbox = video_bboxes[0]
        else:
            screen_bbox = None

        if not annot_position or screen_bbox is None:
            if screen_bbox is None:
                no_bbox += 1
            img.save(out_path)
            saved += 1
            continue

        # Compute where the pure cropped region sits in the full frame
        pure_region, pure_size = compute_pure_region_in_full_frame(img, screen_bbox)

        # Map annotation boxes to full frame coordinates
        boxes = annot_to_fullframe_boxes(annot_position, pure_region, pure_size)

        # Clamp to image bounds
        W_full, H_full = img.size
        clamped = []
        for (l, t, r, b) in boxes:
            clamped.append((
                max(0, min(W_full - 1, l)),
                max(0, min(H_full - 1, t)),
                max(0, min(W_full - 1, r)),
                max(0, min(H_full - 1, b)),
            ))

        annotated = draw_boxes_on_image(img, clamped, width=args.line_width)
        annotated.save(out_path)
        annotated_count += 1
        saved += 1

    print(f"[DONE] total_images={len(all_images)}, saved={saved}, with_boxes={annotated_count}, no_bbox={no_bbox}")


if __name__ == "__main__":
    main()
