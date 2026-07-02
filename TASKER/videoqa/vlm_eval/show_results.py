#!/usr/bin/env python3
"""
Summarise all benchmark results written under the results directory.

Usage:
    python show_results.py
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_BASE_DIR

METHOD_DISPLAY_NAMES = {
    "textonly_0frames": "Text-only (blind)",
    "uniform_16frames": "Uniform (16 frames)",
    "videotree_clip_16frames": "VideoTree-CLIP (16 frames)",
    "videoagent_16frames": "VideoAgent (16 frames)",
    "tasker_16frames": "TASKER (16 frames)",
}


def get_display_name(name):
    return METHOD_DISPLAY_NAMES.get(name, name)


def load_result(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    if not os.path.exists(RESULTS_BASE_DIR):
        print(f"Results directory does not exist: {RESULTS_BASE_DIR}")
        sys.exit(1)

    print("=" * 70)
    print("  Benchmark results summary")
    print(f"  Directory: {RESULTS_BASE_DIR}")
    print("=" * 70)

    results_table = []
    for method_dir in sorted(Path(RESULTS_BASE_DIR).iterdir()):
        if not method_dir.is_dir() or method_dir.name.endswith("_checkpoint"):
            continue
        for dataset_dir in sorted(method_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            result_file = dataset_dir / "results.json"
            if not result_file.exists():
                continue
            result = load_result(result_file)
            if "error" in result:
                print(f"  [WARN] {method_dir.name}/{dataset_dir.name}: {result['error']}")
                continue
            results_table.append({
                "method": method_dir.name,
                "dataset": dataset_dir.name,
                "result": result,
            })

    if not results_table:
        print("\n  No result files found yet.")
        sys.exit(0)

    # EgoSchema
    ego_results = [r for r in results_table if "egoschema" in r["dataset"]]
    if ego_results:
        print("\n" + "-" * 70)
        print("  EgoSchema")
        print("-" * 70)
        print(f"  {'Method':<40} {'Split':<8} {'Acc':<12} {'AvgFrames':<10}")
        for r in ego_results:
            res = r["result"]
            method = get_display_name(r["method"])
            split = res.get("split", r["dataset"])
            acc = res.get("accuracy_pct", "N/A")
            avg_frames = res.get("avg_frames", "N/A")
            if isinstance(avg_frames, (float, int)):
                avg_frames = f"{avg_frames:.1f}"
            print(f"  {method:<40} {split:<8} {acc:<12} {avg_frames:<10}")

    # NExT-QA
    nqa_results = [r for r in results_table if "nextqa" in r["dataset"]]
    if nqa_results:
        print("\n" + "-" * 70)
        print("  NExT-QA")
        print("-" * 70)
        print(f"  {'Method':<40} {'Avg':<8} {'Tem.':<8} {'Cau.':<8} {'Des.':<8} {'Frames':<8}")
        for r in nqa_results:
            res = r["result"]
            method = get_display_name(r["method"])
            avg = res.get("Avg", "N/A")
            by_cat = res.get("by_category", {})
            tem = by_cat.get("Temporal", {}).get("accuracy_pct") or res.get("Temporal", "N/A")
            cau = by_cat.get("Causal", {}).get("accuracy_pct") or res.get("Causal", "N/A")
            des = by_cat.get("Descriptive", {}).get("accuracy_pct") or res.get("Descriptive", "N/A")
            avg_frames = res.get("avg_frames", "N/A")
            if isinstance(avg_frames, (float, int)):
                avg_frames = f"{avg_frames:.1f}"
            print(f"  {method:<40} {avg:<8} {tem:<8} {cau:<8} {des:<8} {avg_frames:<8}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
