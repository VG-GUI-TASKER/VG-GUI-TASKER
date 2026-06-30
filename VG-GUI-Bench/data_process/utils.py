import numpy as np
import os
import subprocess
from PIL import Image


def format_seconds(seconds):
    """Format seconds into HH:MM:SS.mmm for ffmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"


def extract_single_frame(video_path, timestamp, output_path):
    """Extract a single frame at a specific timestamp using ffmpeg."""
    ts_str = format_seconds(timestamp)
    cmd = [
        "ffmpeg",
        "-ss", ts_str,
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_frames(video_path, timestamps, output_dir, prefix="frame", verbose=False):
    """Extract frames from a video at specific timestamps using ffmpeg.
    
    Returns list of saved file paths (temp files ending with _temp.png).
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for i, sec in enumerate(timestamps):
        output_path = os.path.join(output_dir, f"{prefix}_{i:04d}_temp.png")
        final_path = output_path.replace("_temp.png", ".png")

        if os.path.exists(final_path):
            saved.append(final_path)
            continue
        if os.path.exists(output_path):
            saved.append(output_path)
            continue

        if verbose:
            print(f"  Extracting frame {i}/{len(timestamps)} at {sec:.3f}s")

        extract_single_frame(video_path, sec, output_path)
        if os.path.exists(output_path):
            saved.append(output_path)
    return saved


def get_pure_size(image_np: np.ndarray):
    """Compute the bounding box of non-black content in an image.
    
    Returns (x_ratio, y_ratio, w_ratio, h_ratio) in [0,1].
    """
    H, W = image_np.shape[:2]

    if image_np.ndim == 3:
        mask = image_np.max(axis=-1) > 0
    else:
        mask = image_np > 0

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return 0.0, 0.0, 1.0, 1.0

    if coords.shape[1] != 2:
        raise ValueError(f"Unexpected coordinates shape: {coords.shape}")

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    return y0 / W, x0 / H, (y1 - y0) / W, (x1 - x0) / H


def crop_and_save_images(output_dir, crop_boxes):
    """Crop _temp.png images in output_dir using bounding boxes, then refine to pure content.
    
    Saves final images as .png (without _temp suffix) and removes temp files.
    """
    image_paths = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("_temp.png")
    ])

    if len(image_paths) == 0:
        return

    if len(image_paths) != len(crop_boxes):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images vs {len(crop_boxes)} crop boxes"
        )

    for idx, (img_path, box) in enumerate(zip(image_paths, crop_boxes)):
        image = Image.open(img_path)
        cropped_image = image.crop(box)

        frame = np.array(cropped_image)
        H, W, _ = frame.shape
        p_x, p_y, p_w, p_h = get_pure_size(frame)
        x1, y1 = int(p_x * W), int(p_y * H)
        x2, y2 = int((p_x + p_w) * W), int((p_y + p_h) * H)
        pure_image = cropped_image.crop((x1, y1, x2, y2))
        pure_image.save(img_path.replace("_temp.png", ".png"))
        os.remove(img_path)


def crop_single_image(img_path, box):
    """Crop a single image using a bounding box, refine to remove black borders, save in-place."""
    image = Image.open(img_path)
    cropped = image.crop(box)

    frame = np.array(cropped)
    H, W, _ = frame.shape
    p_x, p_y, p_w, p_h = get_pure_size(frame)
    x1, y1 = int(p_x * W), int(p_y * H)
    x2, y2 = int((p_x + p_w) * W), int((p_y + p_h) * H)
    pure = cropped.crop((x1, y1, x2, y2))
    pure.save(img_path)


def rename_temp_to_final(output_dir):
    """Rename all _temp.png files to .png (no crop)."""
    for f in os.listdir(output_dir):
        if f.endswith("_temp.png"):
            src = os.path.join(output_dir, f)
            dst = src.replace("_temp.png", ".png")
            os.rename(src, dst)


def get_video_duration(video_path):
    """Get the duration of a video file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(output.strip())
    except Exception as e:
        print(f"FFprobe Error on {video_path}: {e}")
        return None
