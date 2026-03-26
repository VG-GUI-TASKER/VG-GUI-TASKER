"""
Extract frames at scene timestamps from MONDAY dataset videos.

Modes:
  - default (cut):  extract frames -> crop to screen_bboxes -> remove black borders
  - --no_cut:       extract frames -> save as-is (no cropping)

Supports filtering to only process videos that exist locally in --video_dir.
"""
from datasets import load_dataset
import os
import concurrent.futures
import time
from functools import partial
import argparse
from tqdm import tqdm

from data_process.utils import extract_frames, crop_and_save_images, rename_temp_to_final


def parse_args():
    parser = argparse.ArgumentParser(description="Extract scene-timestamp frames from MONDAY videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing .mp4 video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted images")
    parser.add_argument("--no_cut", action="store_true", help="Skip cropping (save full frames)")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def process_row(my_row, video_dir, output_dir, no_cut=False, verbose=False):
    """Process a single row: extract frames at scene timestamps."""
    try:
        video_id = my_row["video_id"]
        video_file = os.path.join(video_dir, video_id + ".mp4")
        if not os.path.isfile(video_file):
            return False

        scene_timestamps = my_row["scene_timestamps_in_sec"]
        screen_bboxes = my_row["screen_bboxes"]
        output_folder = os.path.join(output_dir, video_id)
        os.makedirs(output_folder, exist_ok=True)

        extract_frames(video_file, scene_timestamps, output_folder, prefix="frame", verbose=verbose)

        if no_cut:
            rename_temp_to_final(output_folder)
        else:
            crop_and_save_images(output_folder, screen_bboxes)

        if verbose:
            print(f"Processed {video_id} finished.")
        return True
    except Exception as e:
        print(f"Error processing {my_row.get('video_id', 'unknown')}: {e}")
        return False


def process_split(dataset, video_dir, output_dir, no_cut, num_workers, verbose):
    process_fn = partial(process_row, video_dir=video_dir, output_dir=output_dir, no_cut=no_cut, verbose=verbose)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_fn, dataset),
            total=len(dataset),
            desc="Processing",
        ))
    return sum(results)


def main():
    args = parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect local video IDs for filtering
    local_video_ids = set(
        os.path.splitext(f)[0]
        for f in os.listdir(args.video_dir)
        if f.endswith(".mp4") and not f.startswith(".")
    )
    print(f"Found {len(local_video_ids)} local MP4 videos in {args.video_dir}")

    print("Loading MONDAY from HuggingFace Hub...")
    dataset_dict = load_dataset("runamu/MONDAY")
    start_time = time.time()

    total_processed = 0
    for split_name in dataset_dict.keys():
        dataset = dataset_dict[split_name]
        # Filter to only rows whose video_id exists locally
        filtered = [row for row in dataset if row["video_id"] in local_video_ids]
        print(f"Split '{split_name}': {len(filtered)}/{len(dataset)} videos found locally")
        if not filtered:
            continue
        processed = process_split(
            filtered, args.video_dir, args.output_dir,
            args.no_cut, args.workers, args.verbose,
        )
        total_processed += processed
        print(f"  Done: {processed}/{len(filtered)} items")

    elapsed = time.time() - start_time
    print(f"Total processed: {total_processed} items in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
