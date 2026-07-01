#!/usr/bin/env python3
"""
VG-GUI-Bench model connectivity check.

Verifies that the configured OpenAI-compatible VLM endpoint can be reached and
handles the three call patterns used by the benchmark:
  1. text-only
  2. single image
  3. multiple images (as in the uniform10 protocol)

Configure the endpoint via environment variables, then run::

    export OPENAI_API_KEY="sk-..."
    export OPENAI_MODEL="gpt-4o"
    # optional, for a local / self-hosted server:
    # export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"

    python test_models.py                       # use env vars
    python test_models.py --model gpt-4o        # override model name
    python test_models.py --dataset-root ./MONDAY
"""

import os
import sys
import glob
import time
import argparse
import traceback

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.model import build_model


TEST_QUESTION = "Describe what you see in this screenshot in one sentence."
TEST_SYSTEM_PROMPT = "You are a GUI automation assistant. Respond concisely."


def find_test_images(dataset_root, count=1):
    """Grab a few sample images from the dataset (or the current directory)."""
    img_dir = os.path.join(dataset_root, "images", "origin")
    imgs = []
    if os.path.isdir(img_dir):
        imgs = sorted(glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True))
    if not imgs:
        for ext in ("*.png", "*.jpg"):
            imgs = sorted(glob.glob(ext))
            if imgs:
                break
    if not imgs:
        return None
    if len(imgs) >= count:
        return imgs[:count]
    return (imgs * ((count // len(imgs)) + 1))[:count]


def print_result(test_name, success, response=None, error=None, elapsed=0.0):
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {test_name} ({elapsed:.1f}s)")
    if success and response:
        preview = str(response)[:100].replace("\n", " ")
        print(f"        response: {preview}...")
    if error:
        print(f"        error: {error}")


def run_check(model, dataset_root):
    all_pass = True

    # Test 1: text only
    try:
        t0 = time.time()
        resp = model(img_path_or_list=None, question="Say hello in one word.")
        ok = bool(resp)
        print_result("text-only", ok, resp, elapsed=time.time() - t0)
        all_pass &= ok
    except Exception as e:  # noqa: BLE001
        print_result("text-only", False, error=str(e))
        all_pass = False

    # Test 2: single image
    imgs = find_test_images(dataset_root, count=1)
    if imgs:
        try:
            t0 = time.time()
            resp = model(img_path_or_list=imgs[0], question=TEST_QUESTION,
                         system_prompt=TEST_SYSTEM_PROMPT, image_first=True)
            ok = bool(resp)
            print_result("single-image", ok, resp, elapsed=time.time() - t0)
            all_pass &= ok
        except Exception as e:  # noqa: BLE001
            print_result("single-image", False, error=str(e))
            all_pass = False
    else:
        print("  [SKIP] image tests (no sample image found)")

    # Test 3: multiple images
    multi = find_test_images(dataset_root, count=3)
    if multi:
        try:
            t0 = time.time()
            resp = model(img_path_or_list=multi, question=TEST_QUESTION,
                         system_prompt=TEST_SYSTEM_PROMPT, image_first=True)
            ok = bool(resp)
            print_result("multi-image (3)", ok, resp, elapsed=time.time() - t0)
            all_pass &= ok
        except Exception as e:  # noqa: BLE001
            print_result("multi-image (3)", False, error=str(e))
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Model connectivity check")
    parser.add_argument("--model", default=None, help="Model name (defaults to OPENAI_MODEL)")
    parser.add_argument("--api_key", default=None, help="API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--base_url", default=None, help="Base URL (defaults to OPENAI_BASE_URL)")
    parser.add_argument("--dataset-root", default="./MONDAY", help="Dataset root for sample images")
    args = parser.parse_args()

    print("=" * 60)
    print("  VG-GUI-Bench model connectivity check")
    print("=" * 60)

    try:
        model = build_model(model_name=args.model, api_key=args.api_key,
                            base_url=args.base_url, max_try=2, timeout=120,
                            max_tokens=256)
    except Exception as e:  # noqa: BLE001
        print(f"  Failed to initialise model: {e}")
        traceback.print_exc()
        return 1

    print(f"  Model: {model}")
    ok = run_check(model, args.dataset_root)

    print()
    if ok:
        print("  All checks passed. You can start run_leaderboard.sh.")
    else:
        print("  Some checks failed. Please fix the errors before evaluating.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
