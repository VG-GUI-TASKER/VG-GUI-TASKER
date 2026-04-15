"""
Desktop GUI benchmark evaluation logic.

Ground-truth action dict schema
--------------------------------
Each step's GT is a list of dicts (multiple valid actions allowed).
Each dict must contain an "action_type" key (string) plus type-specific fields:

  CLICK / DOUBLE_CLICK / RIGHT_CLICK:
    { "action_type": "CLICK",
      "point": [x, y],           # normalised centre of target element
      "bbox":  [x, y, w, h] }    # optional annotated bounding box

  TYPE:
    { "action_type": "TYPE",
      "text": "hello world" }

  HOTKEY:
    { "action_type": "HOTKEY",
      "key": "ctrl+s" }          # normalised inside matching logic

  SCROLL:
    { "action_type": "SCROLL",
      "position":  [x, y],       # cursor position when scrolling
      "direction": "down",        # up | down | left | right
      "bbox": [x, y, w, h] }     # optional scrollable-region bbox

  DRAG:
    { "action_type": "DRAG",
      "start":      [x1, y1],
      "end":        [x2, y2],
      "start_bbox": [x, y, w, h],  # optional
      "end_bbox":   [x, y, w, h] } # optional

  FINISH:
    { "action_type": "FINISH" }
"""

import csv
import datetime
import json
import logging
import math
import re
from collections import defaultdict
from typing import Optional

from desktop_core import action_matching
from desktop_core.action_type import ALL_ACTION_TYPES

# ── Scoring weights ───────────────────────────────────────────────────────────
W_TYPE  = 0.3   # weight for getting the action type correct
W_PARAM = 0.7   # weight for getting the parameters correct


# ══════════════════════════════════════════════════════════════════════════════
# 1. Parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_model_response(response: str) -> tuple[str, dict]:
    """
    Parse a model-generated function-call string into (action_type, params).

    Supported formats
    -----------------
    CLICK(x, y)
    DOUBLE_CLICK(x, y)
    RIGHT_CLICK(x, y)
    TYPE("text")
    HOTKEY("key")
    SCROLL(x, y, "direction")
    DRAG(x1, y1, x2, y2)
    FINISH()

    Returns ("UNKNOWN", {}) when no pattern matches.
    """
    if not response:
        return "NULL", {}
    resp = response.strip()

    # CLICK / DOUBLE_CLICK / RIGHT_CLICK  ─── (x, y)
    for atype in ("DOUBLE_CLICK", "RIGHT_CLICK", "CLICK"):
        m = re.search(
            rf"{atype}\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)",
            resp, re.IGNORECASE
        )
        if m:
            return atype, {"point": [float(m.group(1)), float(m.group(2))]}

    # TYPE("text")
    m = re.search(r'TYPE\s*\(\s*["\']?(.*?)["\']?\s*\)', resp, re.IGNORECASE | re.DOTALL)
    if m:
        return "TYPE", {"text": m.group(1)}

    # HOTKEY("key")
    m = re.search(r'HOTKEY\s*\(\s*["\']?(.*?)["\']?\s*\)', resp, re.IGNORECASE)
    if m:
        return "HOTKEY", {"key": m.group(1).strip()}

    # SCROLL(x, y, "direction")
    m = re.search(
        r'SCROLL\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*["\']?(\w+)["\']?\s*\)',
        resp, re.IGNORECASE
    )
    if m:
        return "SCROLL", {
            "position":  [float(m.group(1)), float(m.group(2))],
            "direction": m.group(3).lower(),
        }

    # DRAG(x1, y1, x2, y2)
    m = re.search(
        r'DRAG\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
        resp, re.IGNORECASE
    )
    if m:
        return "DRAG", {
            "start": [float(m.group(1)), float(m.group(2))],
            "end":   [float(m.group(3)), float(m.group(4))],
        }

    # FINISH()
    if re.search(r"FINISH\s*\(", resp, re.IGNORECASE):
        return "FINISH", {}

    return "UNKNOWN", {}


def process_ground_truth(action_dict: dict) -> dict:
    """
    Normalise a raw GT action dict into a standard {type, params} form.

    Passes through all type-specific fields unchanged; only the
    action_type string is upper-cased for consistency.
    """
    raw_type = action_dict.get("action_type", "UNKNOWN").upper().strip()
    params = {k: v for k, v in action_dict.items() if k != "action_type"}
    return {"type": raw_type, "params": params}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Step-level evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    step_index: int,
    response: str,
    action_common: list,
) -> dict:
    """
    Score one predicted step against (potentially multiple) GT actions.

    Parameters
    ----------
    step_index  : 0-based index within the episode (for logging).
    response    : raw string output from the model.
    action_common : list of GT action dicts for this step.

    Returns
    -------
    dict with keys:
        score         – weighted combined score in [0, 1]
        type_correct  – bool
        param_score   – float in [0, 1]
        gt_type       – string
        pred_type     – string
        error         – string (empty when no error)
    """
    result = {
        "score": 0.0,
        "type_correct": False,
        "param_score": 0.0,
        "gt_type": "UNKNOWN",
        "pred_type": "UNKNOWN",
        "error": "",
    }

    # ── Parse prediction ──────────────────────────────────────────────────────
    try:
        pred_type, pred_params = parse_model_response(response)
        result["pred_type"] = pred_type
    except Exception as exc:
        result["error"] = f"ParseError: {exc}"
        return result

    if not action_common:
        result["error"] = "empty action_common"
        return result

    # ── Score against each GT option; keep best ───────────────────────────────
    best_score = 0.0
    best_info  = None

    for action_dict in action_common:
        gt = process_ground_truth(action_dict)
        if best_info is None:
            result["gt_type"] = gt["type"]  # record first GT type for reporting

        type_correct  = (pred_type == gt["type"])
        param_score   = 0.0

        if type_correct:
            param_score = action_matching.compute_param_score(
                pred_type, pred_params, gt["params"]
            )

        step_score = (W_TYPE + param_score * W_PARAM) if type_correct else 0.0

        if step_score >= best_score:
            best_score = step_score
            best_info  = {
                "score":        step_score,
                "type_correct": type_correct,
                "param_score":  param_score,
                "gt_type":      gt["type"],
                "pred_type":    pred_type,
            }

    if best_info:
        result.update(best_info)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(results: dict) -> dict:
    """
    Aggregate per-step results for one task into summary metrics.

    Parameters
    ----------
    results : { episode_id: [step_result_dict, ...] }

    Returns
    -------
    Flat metrics dict with overall + per-action-type breakdown.
    """
    metrics     = {"Total_Score": 0.0, "Total_Count": 0, "Type_Acc": 0.0}
    class_stats = defaultdict(lambda: {"count": 0, "type_correct": 0, "score": 0.0})

    total_score       = 0.0
    total_type_correct = 0
    count             = 0

    for episode in results.values():
        for step_res in episode:
            count += 1
            total_score        += step_res["score"]
            if step_res["type_correct"]:
                total_type_correct += 1

            gt_type = step_res["gt_type"]
            class_stats[gt_type]["count"] += 1
            class_stats[gt_type]["score"] += step_res["score"]
            if step_res["type_correct"]:
                class_stats[gt_type]["type_correct"] += 1

    if count > 0:
        metrics["Total_Score"] = total_score / count
        metrics["Type_Acc"]    = total_type_correct / count
    metrics["Total_Count"] = count

    # ── Console summary ───────────────────────────────────────────────────────
    logging.info("\n" + "=" * 70)
    logging.info(f"{'Metric':<25} | {'Value'}")
    logging.info("-" * 40)
    logging.info(f"{'Total Steps':<25} | {count}")
    logging.info(f"{'Overall Score':<25} | {metrics['Total_Score']:.4f}")
    logging.info(f"{'Overall Type Acc':<25} | {metrics['Type_Acc']:.4f}")
    logging.info("=" * 70)

    logging.info("\nBreakdown by Action Type:")
    logging.info("-" * 70)
    logging.info(f"{'Type':<16} | {'N':>5} | {'Score':>10} | {'Type Acc':>10}")
    logging.info("-" * 70)

    for atype in ALL_ACTION_TYPES:
        st  = class_stats.get(atype, {"count": 0, "type_correct": 0, "score": 0.0})
        cnt = st["count"]
        avg = st["score"] / cnt if cnt else 0.0
        acc = st["type_correct"] / cnt if cnt else 0.0
        metrics[f"{atype}_Count"]   = cnt
        metrics[f"{atype}_Score"]   = avg
        metrics[f"{atype}_TypeAcc"] = acc
        logging.info(f"{atype:<16} | {cnt:>5} | {avg:>10.4f} | {acc:>10.4f}")

    logging.info("-" * 70 + "\n")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. Top-level pipeline
# ══════════════════════════════════════════════════════════════════════════════

def summarize_and_save_results(args, predictions: dict, start_timestamp: str,
                               extra_info: Optional[dict] = None):
    """
    Score all predictions, log aggregate metrics, and write JSON + CSV outputs.

    Parameters
    ----------
    args         : argparse.Namespace with .prediction_file_path and .csv_path.
    predictions  : { task: { episode_id: [ {response, action_common, ...}, ... ] } }
    start_timestamp : string timestamp of when evaluation started.
    extra_info   : optional dict with run metadata (ref_mode, ref_imgs_dir, …).
    """
    # ── Save raw predictions ──────────────────────────────────────────────────
    with open(args.prediction_file_path, "w", encoding="utf-8") as fp:
        json.dump(predictions, fp, indent=2, ensure_ascii=False)
    print(f"Predictions saved → {args.prediction_file_path}")

    # ── Score every step ──────────────────────────────────────────────────────
    results = {}
    for task, episodes in predictions.items():
        results[task] = {}
        for ep_id, steps in episodes.items():
            results[task][ep_id] = []
            for j, step in enumerate(steps):
                out = evaluate(j, step["response"], step["action_common"])
                results[task][ep_id].append(out)

    # ── Per-task metrics ──────────────────────────────────────────────────────
    eval_dict = {}
    for task in results:
        logging.info("=" * 70)
        logging.info(f"Task: {task}")
        eval_dict[task] = calculate_metrics(results[task])

    avg_score = (
        sum(x["Total_Score"] for x in eval_dict.values()) / len(eval_dict)
        if eval_dict else 0.0
    )
    logging.info(f"[Overall Avg Score]: {avg_score:.4f}")

    # ── Timing & episode completion ───────────────────────────────────────────
    end_ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    t_start = None
    for fmt in ("%Y%m%d%H%M%S", "%Y_%m_%d-%H_%M_%S"):
        try:
            t_start = datetime.datetime.strptime(start_timestamp, fmt)
            break
        except ValueError:
            pass
    t_end   = datetime.datetime.strptime(end_ts, "%Y%m%d%H%M%S")
    elapsed = int((t_end - t_start).total_seconds()) if t_start else 0

    completion_rates = []
    image_counts     = []
    for task, episodes in predictions.items():
        for ep_id, steps in episodes.items():
            ep_res     = results.get(task, {}).get(ep_id, [])
            n_total    = len(ep_res)
            n_correct  = sum(1 for r in ep_res if r["score"] > 0)
            completion_rates.append(n_correct / n_total if n_total else 0.0)
            for step in steps:
                image_counts.append(step.get("num_images", 1))

    avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
    avg_images     = sum(image_counts) / len(image_counts) if image_counts else 0.0

    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 70)
    logging.info(f"  Start Time             : {start_timestamp}")
    logging.info(f"  End Time               : {end_ts}")
    logging.info(f"  Elapsed                : {str(datetime.timedelta(seconds=elapsed))}")
    if extra_info:
        for k, v in extra_info.items():
            logging.info(f"  {k:<23}: {v}")
    logging.info(f"  Avg Episode Completion : {avg_completion:.4f}  ({len(completion_rates)} episodes)")
    logging.info(f"  Avg Images / Step      : {avg_images:.2f}  ({len(image_counts)} steps)")
    logging.info("=" * 70)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.csv_path, "w", newline="", encoding="utf-8") as fp:
        all_keys   = set()
        for m in eval_dict.values():
            all_keys.update(m.keys())
        fieldnames = ["Task"] + sorted(all_keys) + list(vars(args).keys())
        writer     = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for task, m in eval_dict.items():
            writer.writerow({"Task": task, **m, **vars(args)})
    print(f"Metrics saved → {args.csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. History formatting
# ══════════════════════════════════════════════════════════════════════════════

def format_history_action(step_data: dict) -> Optional[str]:
    """
    Convert a stored step dict back to a Function Call string for prompt history.

    step_data should contain an "action_type" string (upper-case) and the
    type-specific parameter fields used during annotation.

    Returns None for unrecognised or UNKNOWN actions (caller should skip them).
    """
    atype = step_data.get("action_type", "").upper().strip()

    if atype in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
        pt = step_data.get("point", [0.0, 0.0])
        return f"{atype}({pt[0]:.3f}, {pt[1]:.3f})"

    if atype == "TYPE":
        text = step_data.get("text", "").replace('"', '\\"')
        return f'TYPE("{text}")'

    if atype == "HOTKEY":
        key = action_matching.normalize_hotkey(step_data.get("key", ""))
        return f'HOTKEY("{key}")'

    if atype == "SCROLL":
        pos = step_data.get("position", [0.5, 0.5])
        direction = step_data.get("direction", "down")
        return f'SCROLL({pos[0]:.3f}, {pos[1]:.3f}, "{direction}")'

    if atype == "DRAG":
        s = step_data.get("start", [0.0, 0.0])
        e = step_data.get("end",   [0.0, 0.0])
        return f"DRAG({s[0]:.3f}, {s[1]:.3f}, {e[0]:.3f}, {e[1]:.3f})"

    if atype == "FINISH":
        return "FINISH()"

    return None
