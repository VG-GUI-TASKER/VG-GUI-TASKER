"""
Desktop GUI action matching utilities.

All coordinates are normalized to [0, 1] relative to screen width/height.
Ground-truth bounding boxes use the format [x_left, y_top, width, height].

Replaces the JAX-based mobile matcher with pure Python / numpy,
and extends the action space with DOUBLE_CLICK, RIGHT_CLICK, HOTKEY, DRAG.
"""

import math
import re
from typing import Optional

import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────

# A predicted click is "correct" if it falls inside the annotated bbox OR if
# its Euclidean distance from the GT point is within this fraction of the screen.
CLICK_DISTANCE_THRESHOLD = 0.10   # ≈ 10% of screen diagonal

# Same spatial tolerance applied to each endpoint of a drag, and to the cursor
# position in a scroll action.
DRAG_DISTANCE_THRESHOLD  = 0.10
SCROLL_DISTANCE_THRESHOLD = 0.12

# Augment annotated bboxes before hit-testing (compensates for annotation slop).
BBOX_WIDTH_AUGMENT  = 1.3
BBOX_HEIGHT_AUGMENT = 1.3

# Modifier key canonical ordering for hotkey normalisation (alphabetical).
_MODIFIERS = ("alt", "ctrl", "meta", "shift")


# ── Hotkey utilities ──────────────────────────────────────────────────────────

def normalize_hotkey(key: str) -> str:
    """
    Canonicalise a hotkey string so that equivalent representations match.

    Rules
    -----
    * Everything lowercased.
    * Modifier keys (alt, ctrl, meta, shift) sorted alphabetically before
      the main key.
    * Parts joined with "+".

    Examples
    --------
    >>> normalize_hotkey("Ctrl+Z")       → "ctrl+z"
    >>> normalize_hotkey("Shift+Ctrl+Z") → "ctrl+shift+z"
    >>> normalize_hotkey("ENTER")        → "enter"
    """
    parts = [p.strip().lower() for p in key.split("+") if p.strip()]
    mods  = sorted(p for p in parts if p in _MODIFIERS)
    main  = [p for p in parts if p not in _MODIFIERS]
    return "+".join(mods + main)


# ── Bbox utilities ────────────────────────────────────────────────────────────

def _augment_bbox(bbox: list, w_frac: float, h_frac: float) -> list:
    """
    Expand bbox by given fractions symmetrically.

    bbox format: [x_left, y_top, width, height]  (all in [0, 1])
    """
    x, y, w, h = bbox
    new_w = w * w_frac
    new_h = h * h_frac
    return [
        max(0.0, x - (new_w - w) / 2),
        max(0.0, y - (new_h - h) / 2),
        min(1.0, new_w),
        min(1.0, new_h),
    ]


def point_in_bbox(
    point: list,
    bbox: list,
    w_augment: float = BBOX_WIDTH_AUGMENT,
    h_augment: float = BBOX_HEIGHT_AUGMENT,
) -> bool:
    """
    Return True if *point* [x, y] falls inside the (augmented) *bbox*.

    bbox format: [x_left, y_top, width, height]  (all in [0, 1])
    """
    if not bbox or len(bbox) < 4:
        return False
    ax, ay, aw, ah = _augment_bbox(bbox, w_augment, h_augment)
    px, py = point
    return (ax <= px <= ax + aw) and (ay <= py <= ay + ah)


def points_match(
    pred: list,
    gt: list,
    gt_bbox: Optional[list] = None,
    threshold: float = CLICK_DISTANCE_THRESHOLD,
) -> bool:
    """
    Return True if *pred* [x, y] is considered to hit the same target as *gt*.

    Accepts if either:
      (a) *pred* falls inside the augmented *gt_bbox*, OR
      (b) Euclidean distance between *pred* and *gt* ≤ *threshold*.
    """
    if gt_bbox and point_in_bbox(pred, gt_bbox):
        return True
    return math.dist(pred, gt) <= threshold


# ── Per-action matching ───────────────────────────────────────────────────────

def match_click(pred_params: dict, gt_params: dict) -> float:
    """
    Score for CLICK / DOUBLE_CLICK / RIGHT_CLICK (spatial matching only).

    Returns 1.0 on hit, 0.0 otherwise.
    """
    pred_pt = pred_params.get("point")
    gt_pt   = gt_params.get("point")
    if pred_pt is None or gt_pt is None:
        return 0.0
    gt_bbox = gt_params.get("bbox")
    return 1.0 if points_match(pred_pt, gt_pt, gt_bbox, CLICK_DISTANCE_THRESHOLD) else 0.0


def match_type(pred_params: dict, gt_params: dict, threshold: float = 0.8) -> float:
    """
    Score for TYPE: normalised Levenshtein similarity ≥ threshold → full score.

    Returns the similarity value if ≥ threshold, else 0.0.
    """
    pred_text = str(pred_params.get("text", "")).strip().lower()
    gt_text   = str(gt_params.get("text", "")).strip().lower()
    if not gt_text and not pred_text:
        return 1.0
    if not gt_text or not pred_text:
        return 0.0
    sim = _text_similarity(pred_text, gt_text)
    return sim if sim >= threshold else 0.0


def match_hotkey(pred_params: dict, gt_params: dict) -> float:
    """
    Score for HOTKEY: exact match after normalisation.

    Returns 1.0 on match, 0.0 otherwise.
    """
    pred_key = normalize_hotkey(pred_params.get("key", ""))
    gt_key   = normalize_hotkey(gt_params.get("key", ""))
    return 1.0 if pred_key == gt_key else 0.0


def match_scroll(pred_params: dict, gt_params: dict) -> float:
    """
    Score for SCROLL.

    Scoring breakdown (total 1.0):
      0.5  — direction matches (up / down / left / right)
      0.5  — cursor position (x, y) is within SCROLL_DISTANCE_THRESHOLD of GT

    A wrong direction scores 0.0 regardless of position.
    """
    pred_dir = pred_params.get("direction", "").lower().strip()
    gt_dir   = gt_params.get("direction", "").lower().strip()

    if pred_dir != gt_dir:
        return 0.0   # wrong direction → no credit

    score = 0.5      # direction correct

    pred_pos = pred_params.get("position")
    gt_pos   = gt_params.get("position")
    gt_bbox  = gt_params.get("bbox")   # optional: scrollable region bbox

    if pred_pos and gt_pos:
        if points_match(pred_pos, gt_pos, gt_bbox, SCROLL_DISTANCE_THRESHOLD):
            score += 0.5

    return score


def match_drag(pred_params: dict, gt_params: dict) -> float:
    """
    Score for DRAG.

    Scoring breakdown (total 1.0):
      0.5  — start point matches
      0.5  — end point matches

    Each endpoint is checked against its respective GT position + optional bbox.
    """
    score = 0.0

    pred_start = pred_params.get("start")
    gt_start   = gt_params.get("start")
    if pred_start and gt_start:
        bbox_start = gt_params.get("start_bbox")
        if points_match(pred_start, gt_start, bbox_start, DRAG_DISTANCE_THRESHOLD):
            score += 0.5

    pred_end = pred_params.get("end")
    gt_end   = gt_params.get("end")
    if pred_end and gt_end:
        bbox_end = gt_params.get("end_bbox")
        if points_match(pred_end, gt_end, bbox_end, DRAG_DISTANCE_THRESHOLD):
            score += 0.5

    return score


# ── Unified entry point ───────────────────────────────────────────────────────

def compute_param_score(
    action_type: str,
    pred_params: dict,
    gt_params: dict,
) -> float:
    """
    Dispatch to the appropriate per-action matcher.

    Parameters
    ----------
    action_type : str
        One of "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "TYPE",
        "HOTKEY", "SCROLL", "DRAG", "FINISH".
    pred_params, gt_params : dict
        Parsed parameter dicts for the predicted and ground-truth actions.

    Returns
    -------
    float
        Parameter score in [0, 1].
    """
    if action_type in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
        return match_click(pred_params, gt_params)
    elif action_type == "TYPE":
        return match_type(pred_params, gt_params)
    elif action_type == "HOTKEY":
        return match_hotkey(pred_params, gt_params)
    elif action_type == "SCROLL":
        return match_scroll(pred_params, gt_params)
    elif action_type == "DRAG":
        return match_drag(pred_params, gt_params)
    elif action_type == "FINISH":
        return 1.0   # no params to score
    else:
        return 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _text_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    dist = _levenshtein(s1, s2)
    return 1.0 - dist / max(len(s1), len(s2))
