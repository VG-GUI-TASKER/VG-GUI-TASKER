"""
Desktop GUI benchmark action type definitions.
Designed for VG-GUI-Bench desktop pipeline.
"""

import enum


class ActionType(enum.IntEnum):
    """Integer values for each supported action type in desktop GUI evaluation."""

    # ── Pointer actions ──────────────────────────────────────────────────────
    # Left single click at a normalized (x, y) coordinate.
    CLICK = 0

    # Left double-click at a normalized (x, y) coordinate.
    # Used for: opening files/folders, entering cell edit mode, selecting words.
    DOUBLE_CLICK = 1

    # Right click at a normalized (x, y) coordinate.
    # Used for: triggering context menus, which are pervasive in desktop software.
    RIGHT_CLICK = 2

    # ── Keyboard actions ─────────────────────────────────────────────────────
    # Type text into the currently focused input field.
    # Does NOT include implicit focus click or trailing Enter.
    TYPE = 3

    # Execute a keyboard shortcut, e.g. "ctrl+s", "ctrl+z", "f2", "escape".
    # Modifier keys are normalized: lowercase, sorted alt < ctrl < meta < shift.
    HOTKEY = 4

    # ── Compound pointer actions ─────────────────────────────────────────────
    # Scroll the mouse wheel at position (x, y) in a given direction.
    # direction ∈ {"up", "down", "left", "right"}
    # Covers: scrolling document panels, timeline, canvas, property lists.
    SCROLL = 5

    # Click-and-drag from (x1, y1) to (x2, y2).
    # Covers: moving timeline clips, resizing columns, rotating viewports (Blender),
    #         painting brush strokes (Photoshop), adjusting sliders, reordering layers.
    DRAG = 6

    # ── Episode status ───────────────────────────────────────────────────────
    # Task is complete, or cannot be continued (impossible / already done).
    FINISH = 10


# ── Human-readable labels ─────────────────────────────────────────────────────
ACTION_TYPE_TO_STR = {
    ActionType.CLICK:        "CLICK",
    ActionType.DOUBLE_CLICK: "DOUBLE_CLICK",
    ActionType.RIGHT_CLICK:  "RIGHT_CLICK",
    ActionType.TYPE:         "TYPE",
    ActionType.HOTKEY:       "HOTKEY",
    ActionType.SCROLL:       "SCROLL",
    ActionType.DRAG:         "DRAG",
    ActionType.FINISH:       "FINISH",
}

STR_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_TO_STR.items()}

ALL_ACTION_TYPES = list(ACTION_TYPE_TO_STR.values())
