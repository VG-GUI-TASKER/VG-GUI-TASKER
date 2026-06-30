"""
Extract N uniformly-spaced frames from MONDAY dataset videos.

Modes:
  - default (cut):  extract frames -> crop using first screen_bbox from HF dataset -> remove black borders
  - --no_cut:       extract frames -> save as-is (no cropping)
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset

from data_process.utils import (
    extract_single_frame,
    get_video_duration,
    crop_single_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract N uniform frames from MONDAY videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing .mp4 video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for saving output images")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of uniformly-spaced frames to extract")
    parser.add_argument("--no_cut", action="store_true", help="Skip cropping (save full frames)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    return parser.parse_args()


def process_video(video_id, video_dir, output_root, num_frames, bbox, no_cut):
    """Worker function to process a single video."""
    video_path = os.path.join(video_dir, video_id + ".mp4")
    if not os.path.isfile(video_path):
        return "SKIP"

    duration = get_video_duration(video_path)
    if duration is None:
        return "ERROR"

    safe_end = max(0, duration - 0.1)
    timestamps = np.linspace(0, safe_end, num_frames).tolist()

    save_dir = os.path.join(output_root, video_id)
    os.makedirs(save_dir, exist_ok=True)

    for i, ts in enumerate(timestamps):
        out_path = os.path.join(save_dir, f"frame_uniform_{i:04d}.png")
        if os.path.exists(out_path):
            continue
        extract_single_frame(video_path, ts, out_path)

        if (not no_cut) and bbox is not None and os.path.exists(out_path):
            try:
                crop_single_image(out_path, bbox)
            except Exception as e:
                print(f"Crop error {video_id} frame {i}: {e}")

    return "SUCCESS"


def main():
    args = parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"Error: Directory not found: {args.video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Scan local video files
    all_files = os.listdir(args.video_dir)
    video_ids = [os.path.splitext(f)[0] for f in all_files if f.endswith(".mp4") and not f.startswith(".")]
    print(f"Found {len(video_ids)} MP4 videos locally.")

    # Build video_id -> first screen_bbox mapping from HF dataset (only needed for crop mode)
    bbox_map = {}
    if not args.no_cut:
        local_set = set(video_ids)
        print("Loading MONDAY from HuggingFace Hub for screen_bboxes...")
        dataset_dict = load_dataset("runamu/MONDAY")
        for split_name in dataset_dict.keys():
            for row in dataset_dict[split_name]:
                vid = row["video_id"]
                if vid in local_set and vid not in bbox_map and row["screen_bboxes"]:
                    bbox_map[vid] = row["screen_bboxes"][0]  # Use first bbox
        print(f"  Got screen_bboxes for {len(bbox_map)}/{len(local_set)} local videos.")

    print(f"Extracting {args.num_frames} uniform frames per video, no_cut={args.no_cut}, workers={args.workers}")

    success = 0
    errors = 0
    skipped = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for vid in video_ids:
            bbox = bbox_map.get(vid, None)
            fut = executor.submit(
                process_video, vid, args.video_dir, args.output_dir,
                args.num_frames, bbox, args.no_cut,
            )
            futures[fut] = vid

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result == "SUCCESS":
                success += 1
            elif result == "SKIP":
                skipped += 1
            else:
                errors += 1

    print(f"\nDone. Success: {success}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
