"""
Draw red bounding boxes (annot_position) on MONDAY frames.

Iterates over ALL images found in --images_root (origin/), and for each image
looks up annotation data from ours_data.json. Images without annotation data
are copied as-is, ensuring output count == input count.
"""
import os
import json
import argparse
import math
from typing import List, Tuple, Dict

from PIL import Image, ImageDraw
from tqdm import tqdm


def annot_to_boxes(annot_position: List[float], W: int, H: int) -> List[Tuple[int, int, int, int]]:
    """
    annot_position: every 4 numbers = (y_top_left, x_top_left, box_height, box_width) in [0,1].
    Returns list of (left, top, right, bottom) in pixels.
    """
    if not annot_position:
        return []

    usable_len = (len(annot_position) // 4) * 4
    annot_position = annot_position[:usable_len]

    boxes = []
    for i in range(0, usable_len, 4):
        y, x, h, w = annot_position[i:i + 4]
        y = max(0.0, min(1.0, y))
        x = max(0.0, min(1.0, x))
        h = max(0.0, min(1.0, h))
        w = max(0.0, min(1.0, w))

        left = int(math.floor(x * W))
        top = int(math.floor(y * H))
        right = int(math.ceil((x + w) * W))
        bottom = int(math.ceil((y + h) * H))

        left = max(0, min(W - 1, left))
        top = max(0, min(H - 1, top))
        right = max(0, min(W - 1, right))
        bottom = max(0, min(H - 1, bottom))

        if right <= left:
            right = min(W - 1, left + 1)
        if bottom <= top:
            bottom = min(H - 1, top + 1)

        boxes.append((left, top, right, bottom))
    return boxes


def draw_boxes_on_image(img: Image.Image, boxes: List[Tuple[int, int, int, int]], width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for (l, t, r, b) in boxes:
        draw.rectangle([l, t, r, b], outline="red", width=width)
    return out


def build_annot_index(json_path: str) -> Dict[str, list]:
    """Build a mapping: img_filename (e.g. 'SIjOxM9jVj8/frame_0000') -> annot_position list.
    
    Uses the first action's annot_position for each step.
    """
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
    E.g. ('SIjOxM9jVj8', 'frame_0000', '/path/to/origin/SIjOxM9jVj8/frame_0000.png')
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


def parse_args():
    parser = argparse.ArgumentParser(description="Draw red bboxes on MONDAY frames")
    parser.add_argument("--json_path", type=str, required=True, help="Path to ours_data.json")
    parser.add_argument("--images_root", type=str, required=True, help="Root dir of input images (e.g. origin/)")
    parser.add_argument("--out_root", type=str, required=True, help="Root dir to save annotated images")
    parser.add_argument("--line_width", type=int, default=3, help="Rectangle line width")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated images")
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

    saved = 0
    annotated_count = 0

    for video_id, frame_base, img_path in tqdm(all_images, desc="Annotating"):
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

        # Look up annotation by img_filename key (e.g. "SIjOxM9jVj8/frame_0000")
        lookup_key = f"{video_id}/{frame_base}"
        annot_position = annot_index.get(lookup_key, [])

        W, H = img.size
        boxes = annot_to_boxes(annot_position, W, H)

        if boxes:
            annotated = draw_boxes_on_image(img, boxes, width=args.line_width)
            annotated.save(out_path)
            annotated_count += 1
        else:
            img.save(out_path)

        saved += 1

    print(f"[DONE] total_images={len(all_images)}, saved={saved}, with_boxes={annotated_count}")


if __name__ == "__main__":
    main()
