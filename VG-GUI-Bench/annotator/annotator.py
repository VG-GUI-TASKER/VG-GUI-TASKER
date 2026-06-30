"""
VG-GUI-Bench Desktop Annotation Tool
=====================================
Keyboard shortcuts
  1-8  : select action type  (1=CLICK 2=DOUBLE_CLICK 3=RIGHT_CLICK
                               4=SCROLL 5=DRAG 6=TYPE 7=HOTKEY 8=FINISH)
  Space        : play / pause
  Left / Right : step ±1 frame   (Shift: ±10 frames)
  Up / Down    : jump ±5 %
  Enter        : save current video and go to next
  Ctrl+Z       : undo last annotation
  Escape       : cancel current pending action
  Delete / D   : delete selected annotation in list
"""

import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
except ImportError:
    sys.exit("tkinter is required")

# ── Audio (pygame) ────────────────────────────────────────────────────────────
_AUDIO_OK: bool
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False

# ── ffmpeg for audio extraction ───────────────────────────────────────────────
def _find_ffmpeg() -> Optional[str]:
    f = shutil.which("ffmpeg")
    if f:
        return f
    # JianyingPro bundled ffmpeg (common on this machine)
    base = Path(r"C:\Users\liuqi\AppData\Local\JianyingPro\Apps")
    if base.exists():
        for p in sorted(base.iterdir(), reverse=True):
            ff = p / "ffmpeg.exe"
            if ff.exists():
                return str(ff)
    return None

FFMPEG = _find_ffmpeg()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
VIDEO_ROOT = ROOT / "desktop_pipeline" / "video_selected"
INFO_JSON  = VIDEO_ROOT / "info.json"

# ── UI constants ──────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 896, 504
PANEL_W = 380
WIN_W   = CANVAS_W + PANEL_W + 24
WIN_H   = CANVAS_H + 160

BG      = "#1e1e2e"
BG2     = "#2a2a3e"
BG3     = "#313145"
FG      = "#cdd6f4"
FG2     = "#a6adc8"
ACC     = "#89b4fa"
ACC2    = "#cba6f7"
ERR     = "#f38ba8"
OK      = "#a6e3a1"
WARN    = "#f9e2af"
SCROLL_CLR = "#45475a"

FONT_SM  = ("Segoe UI", 9)
FONT_MD  = ("Segoe UI", 10)
FONT_BD  = ("Segoe UI", 10, "bold")
FONT_LG  = ("Segoe UI", 12, "bold")
FONT_MONO= ("Consolas",  9)

ACTION_LABELS = {
    "1": "CLICK",
    "2": "DOUBLE_CLICK",
    "3": "RIGHT_CLICK",
    "4": "SCROLL",
    "5": "DRAG",
    "6": "TYPE",
    "7": "HOTKEY",
    "8": "FINISH",
}
ACTION_COLORS = {
    "CLICK":        "#89b4fa",
    "DOUBLE_CLICK": "#b4befe",
    "RIGHT_CLICK":  "#cba6f7",
    "SCROLL":       "#94e2d5",
    "DRAG":         "#f9e2af",
    "TYPE":         "#a6e3a1",
    "HOTKEY":       "#fab387",
    "FINISH":       "#f38ba8",
}
DIR_KEYS = {"Left": "left", "Right": "right", "Up": "up", "Down": "down"}


# ══════════════════════════════════════════════════════════════════════════════
def _ask_domain(parent: tk.Tk, domains: list[str]) -> Optional[str]:
    """
    Show a modal dialog to pick a starting domain.

    Returns the chosen domain string, or None if the user clicks
    'Auto-resume' (start from the first unannotated video globally).
    """
    result: list[Optional[str]] = [None]

    dlg = tk.Toplevel(parent)
    dlg.title("Select Starting Domain")
    dlg.configure(bg=BG)
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(dlg, text="Start from domain:", bg=BG, fg=FG,
             font=FONT_BD, pady=8, padx=16).pack(anchor="w")
    tk.Label(dlg, text="Choose a domain to begin annotating from its first\n"
                       "unannotated video — useful for dividing work with teammates.",
             bg=BG, fg=FG2, font=FONT_SM, padx=16, justify="left").pack(anchor="w")

    # scrollable list of domains
    frame = tk.Frame(dlg, bg=BG2)
    frame.pack(fill="both", expand=True, padx=16, pady=8)

    sb = ttk.Scrollbar(frame, orient="vertical")
    lb = tk.Listbox(frame, bg=BG3, fg=FG, font=FONT_MD,
                    selectbackground=ACC, selectforeground=BG,
                    relief="flat", highlightthickness=0, bd=0,
                    yscrollcommand=sb.set, height=min(len(domains), 14))
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(fill="both", expand=True)
    for d in domains:
        lb.insert("end", d)
    lb.selection_set(0)

    def _confirm():
        sel = lb.curselection()
        result[0] = domains[sel[0]] if sel else None
        dlg.destroy()

    def _auto():
        result[0] = None
        dlg.destroy()

    btn_row = tk.Frame(dlg, bg=BG)
    btn_row.pack(fill="x", padx=16, pady=(0, 12))
    tk.Button(btn_row, text="Start from selected domain",
              command=_confirm, bg=ACC, fg=BG, font=FONT_BD,
              relief="flat", padx=10, pady=5, bd=0).pack(side="left", padx=(0, 8))
    tk.Button(btn_row, text="Auto-resume",
              command=_auto, bg=BG3, fg=FG2, font=FONT_SM,
              relief="flat", padx=10, pady=5, bd=0).pack(side="left")

    lb.bind("<Double-Button-1>", lambda _: _confirm())
    dlg.bind("<Return>",         lambda _: _confirm())
    dlg.bind("<Escape>",         lambda _: _auto())

    # centre over parent
    parent.update_idletasks()
    px, py = parent.winfo_x(), parent.winfo_y()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    dlg.update_idletasks()
    dw, dh = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw)//2}+{py + (ph - dh)//2}")

    parent.wait_window(dlg)
    return result[0]


# ══════════════════════════════════════════════════════════════════════════════
class AnnotatorApp:
    # ── States ─────────────────────────────────────────────────────────────
    IDLE        = "idle"
    WAIT_DIR    = "wait_dir"       # SCROLL: waiting for arrow key
    DRAW_BBOX   = "draw_bbox"      # drawing first (or only) bbox
    DRAW_BBOX2  = "draw_bbox2"     # DRAG: drawing second bbox
    WAIT_HOTKEY = "wait_hotkey"    # HOTKEY: waiting for key combo

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("VG-GUI-Bench Annotator")
        root.configure(bg=BG)
        root.resizable(False, False)

        # ── Video state ────────────────────────────────────────────────────
        self.cap:        Optional[cv2.VideoCapture] = None
        self._cap_lock   = threading.Lock()
        self.total_frames = 0
        self.fps          = 30.0
        self.cur_frame    = 0
        self.playing      = False
        self._stop_event  = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self._poll_job    = None
        self._photo       = None          # keep reference
        self._speed       = 1.0           # playback speed multiplier

        # ── Audio state ────────────────────────────────────────────────────
        self._audio_file: Optional[str] = None   # path to cached WAV
        self._audio_ready = False                 # extraction done
        self._audio_prep_thread: Optional[threading.Thread] = None

        # display geometry (letterbox)
        self._disp_x = self._disp_y = 0
        self._disp_scale = 1.0
        self._vid_w = self._vid_h = 1

        # ── Annotation state ───────────────────────────────────────────────
        self.annotations: list[dict] = []
        self._undo_stack: list       = []
        self._pending:   dict        = {}
        self._state      = self.IDLE
        self._sel_idx    = -1            # selected annotation in list
        self._updating_timeline = False  # re-entrance guard

        # bbox drawing
        self._bbox_start_canvas = None
        self._bbox_rect_id      = None
        self._bbox_stage        = 0      # 1 or 2 for DRAG

        # ── Video list ─────────────────────────────────────────────────────
        self._video_list: list[dict] = []  # [{domain, idx, path, task, title}, ...]
        self._video_pos  = 0

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_ui()
        self._load_video_list()
        self._load_video(self._video_pos)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = self.root

        # ── Top bar ────────────────────────────────────────────────────────
        top = tk.Frame(root, bg=BG2, height=36)
        top.pack(fill="x", padx=0, pady=0)
        top.pack_propagate(False)

        nav_cfg = dict(bg=BG3, fg=FG2, font=FONT_SM, relief="flat",
                       activebackground=BG, activeforeground=ACC,
                       padx=10, pady=0, bd=0)
        tk.Button(top, text="◀ Prev", command=self._go_prev, **nav_cfg).pack(
            side="left", padx=(4, 0))
        tk.Button(top, text="Next ▶", command=self._go_next, **nav_cfg).pack(
            side="left", padx=(2, 8))

        self._lbl_task = tk.Label(top, text="", bg=BG2, fg=FG,
                                  font=FONT_BD, anchor="w", padx=4)
        self._lbl_task.pack(side="left", fill="x", expand=True)

        self._lbl_progress = tk.Label(top, text="", bg=BG2, fg=FG2,
                                      font=FONT_SM, padx=10)
        self._lbl_progress.pack(side="right")

        # ── Main area ──────────────────────────────────────────────────────
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 0))

        # canvas
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg="#000000", highlightthickness=1,
                                highlightbackground=BG3, cursor="crosshair")
        self.canvas.pack()

        # timeline
        tl_frame = tk.Frame(left, bg=BG2, height=36)
        tl_frame.pack(fill="x", pady=(4, 0))
        tl_frame.pack_propagate(False)

        self._timeline = ttk.Scale(tl_frame, from_=0, to=1000,
                                   orient="horizontal",
                                   command=self._on_timeline_drag)
        self._timeline.pack(side="left", fill="x", expand=True, padx=8, pady=6)
        self._lbl_time = tk.Label(tl_frame, text="0:00 / 0:00",
                                  bg=BG2, fg=FG2, font=FONT_MONO, width=14)
        self._lbl_time.pack(side="right", padx=8)

        # ── Row 1: play + seek buttons ─────────────────────────────────────
        tb = tk.Frame(left, bg=BG)
        tb.pack(fill="x", pady=(4, 0))

        btn_cfg = dict(bg=BG3, fg=FG, font=FONT_SM, relief="flat",
                       activebackground=BG2, activeforeground=ACC,
                       padx=7, pady=4, bd=0)

        self._btn_play = tk.Button(tb, text="▶  Play", command=self._toggle_play,
                                   **btn_cfg)
        self._btn_play.pack(side="left", padx=(0, 6))

        # frame-level step (← →)
        tk.Button(tb, text="◀1f",  command=lambda: self._step(-1),
                  **btn_cfg).pack(side="left", padx=1)
        tk.Button(tb, text="1f▶",  command=lambda: self._step(1),
                  **btn_cfg).pack(side="left", padx=1)

        tk.Label(tb, text="", bg=BG, width=1).pack(side="left")

        # second-level seek (Shift+← →)
        tk.Button(tb, text="◀1s",  command=lambda: self._seek_secs(-1),
                  **btn_cfg).pack(side="left", padx=1)
        tk.Button(tb, text="1s▶",  command=lambda: self._seek_secs(1),
                  **btn_cfg).pack(side="left", padx=1)
        tk.Button(tb, text="◀5s",  command=lambda: self._seek_secs(-5),
                  **btn_cfg).pack(side="left", padx=1)
        tk.Button(tb, text="5s▶",  command=lambda: self._seek_secs(5),
                  **btn_cfg).pack(side="left", padx=1)

        btn_save = tk.Button(tb, text="✔  Save & Next  [Enter]",
                             command=self._save_and_next,
                             bg="#1e3a2f", fg=OK, font=FONT_BD,
                             relief="flat", activebackground="#2a4a3a",
                             activeforeground=OK, padx=10, pady=4, bd=0)
        btn_save.pack(side="right", padx=(8, 0))

        btn_skip = tk.Button(tb, text="Skip",
                             command=self._skip_video,
                             bg=BG3, fg=WARN, font=FONT_SM,
                             relief="flat", activebackground=BG2,
                             activeforeground=WARN, padx=8, pady=4, bd=0)
        btn_skip.pack(side="right", padx=4)

        # ── Row 2: speed selector ──────────────────────────────────────────
        tb2 = tk.Frame(left, bg=BG)
        tb2.pack(fill="x", pady=(2, 0))

        tk.Label(tb2, text="Speed:", bg=BG, fg=FG2, font=FONT_SM).pack(side="left")
        self._speed_btns: dict[float, tk.Button] = {}
        for spd in (1.0, 1.5, 2.0, 3.0):
            lbl = f"{spd:g}x"
            b = tk.Button(tb2, text=lbl,
                          command=lambda s=spd: self._set_speed(s),
                          bg=ACC if spd == 1.0 else BG3,
                          fg=BG if spd == 1.0 else FG2,
                          font=FONT_SM, relief="flat",
                          activebackground=BG2, activeforeground=ACC,
                          padx=7, pady=3, bd=0)
            b.pack(side="left", padx=2)
            self._speed_btns[spd] = b

        tk.Label(tb2, text="  ←/→ frame  |  Shift+←/→ 1s  |  Ctrl+←/→ 5s",
                 bg=BG, fg=SCROLL_CLR, font=FONT_SM).pack(side="left", padx=8)

        # ── Right panel ────────────────────────────────────────────────────
        panel = tk.Frame(main, bg=BG2, width=PANEL_W)
        panel.pack(side="right", fill="both", padx=(8, 0))
        panel.pack_propagate(False)

        # action buttons
        tk.Label(panel, text="Action Type  (1-8)", bg=BG2, fg=FG2,
                 font=FONT_SM).pack(anchor="w", padx=8, pady=(8, 2))

        btn_grid = tk.Frame(panel, bg=BG2)
        btn_grid.pack(fill="x", padx=8)
        self._action_btns: dict[str, tk.Button] = {}
        for i, (key, label) in enumerate(ACTION_LABELS.items()):
            clr = ACTION_COLORS[label]
            b = tk.Button(btn_grid, text=f"{key}  {label}",
                          command=lambda l=label: self._select_action(l),
                          bg=BG3, fg=clr, font=FONT_SM,
                          relief="flat", activebackground=BG,
                          activeforeground=clr, anchor="w",
                          padx=6, pady=3, bd=0)
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=1)
            self._action_btns[label] = b
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # status / hint
        self._lbl_state = tk.Label(panel, text="Select an action type",
                                   bg=BG2, fg=ACC, font=FONT_BD,
                                   wraplength=PANEL_W - 20, justify="left")
        self._lbl_state.pack(anchor="w", padx=8)

        self._lbl_hint = tk.Label(panel, text="",
                                  bg=BG2, fg=FG2, font=FONT_SM,
                                  wraplength=PANEL_W - 20, justify="left")
        self._lbl_hint.pack(anchor="w", padx=8, pady=(2, 0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # annotation list
        tk.Label(panel, text="Annotations", bg=BG2, fg=FG2,
                 font=FONT_SM).pack(anchor="w", padx=8)

        list_frame = tk.Frame(panel, bg=BG2)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(2, 4))

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self._ann_list = tk.Listbox(list_frame, bg=BG3, fg=FG,
                                    font=FONT_MONO, selectbackground=ACC,
                                    selectforeground=BG, relief="flat",
                                    highlightthickness=0, bd=0,
                                    yscrollcommand=scrollbar.set)
        scrollbar.config(command=self._ann_list.yview)
        scrollbar.pack(side="right", fill="y")
        self._ann_list.pack(fill="both", expand=True)
        self._ann_list.bind("<<ListboxSelect>>", self._on_list_select)
        self._ann_list.bind("<Double-Button-1>", self._on_list_goto)

        # bottom buttons
        bb = tk.Frame(panel, bg=BG2)
        bb.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(bb, text="Ctrl+Z  Undo", command=self._undo,
                  bg=BG3, fg=WARN, font=FONT_SM, relief="flat",
                  activebackground=BG, activeforeground=WARN,
                  padx=6, pady=3, bd=0).pack(side="left", padx=(0, 4))
        tk.Button(bb, text="Del  Remove", command=self._delete_selected,
                  bg=BG3, fg=ERR, font=FONT_SM, relief="flat",
                  activebackground=BG, activeforeground=ERR,
                  padx=6, pady=3, bd=0).pack(side="left")

        # ── Key bindings ───────────────────────────────────────────────────
        root.bind("<Key>",          self._on_key)
        root.bind("<Control-z>",    lambda e: self._undo())
        root.bind("<Return>",       lambda e: self._save_and_next())
        root.bind("<Escape>",       lambda e: self._cancel_pending())
        root.bind("<space>",        lambda e: self._toggle_play())
        root.bind("<Left>",          lambda _: self._step(-1))
        root.bind("<Right>",         lambda _: self._step(1))
        root.bind("<Shift-Left>",    lambda _: self._seek_secs(-1))
        root.bind("<Shift-Right>",   lambda _: self._seek_secs(1))
        root.bind("<Control-Left>",  lambda _: self._seek_secs(-5))
        root.bind("<Control-Right>", lambda _: self._seek_secs(5))
        root.bind("<Up>",            lambda _: self._jump_pct(-5))
        root.bind("<Down>",          lambda _: self._jump_pct(5))
        root.bind("<Delete>",       lambda e: self._delete_selected())
        root.bind("d",              lambda e: self._delete_selected())

        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    # ══════════════════════════════════════════════════════════════════════════
    # Video list management
    # ══════════════════════════════════════════════════════════════════════════

    def _load_video_list(self):
        if not INFO_JSON.exists():
            messagebox.showerror("Error", f"info.json not found:\n{INFO_JSON}")
            sys.exit(1)
        with open(INFO_JSON, encoding="utf-8") as f:
            info = json.load(f)

        for domain, items in info.items():
            for item in items:
                rel = item.get("path", "")
                abs_path = VIDEO_ROOT / rel
                if abs_path.exists():
                    self._video_list.append({
                        "domain": domain,
                        "path":   abs_path,
                        "task":   item.get("task", ""),
                        "title":  item.get("title", ""),
                        "id":     item.get("id", ""),
                    })

        if not self._video_list:
            messagebox.showerror("Error", "No video files found under video_selected/")
            sys.exit(1)

        # ── Domain-start dialog ───────────────────────────────────────────
        domains = list(dict.fromkeys(v["domain"] for v in self._video_list))
        chosen = _ask_domain(self.root, domains)
        # chosen is either a domain name (start from that domain) or None (auto-resume)

        if chosen is not None:
            # find first unannotated video in the chosen domain
            for i, v in enumerate(self._video_list):
                if v["domain"] == chosen:
                    ann_path = v["path"].parent / "annotations.json"
                    if not ann_path.exists():
                        self._video_pos = i
                        return
            # domain fully done — fall through to auto-resume
        # auto-resume: first unannotated video overall
        for i, v in enumerate(self._video_list):
            ann_path = v["path"].parent / "annotations.json"
            if not ann_path.exists():
                self._video_pos = i
                return
        self._video_pos = 0

    def _current_video(self) -> dict:
        return self._video_list[self._video_pos]

    # ══════════════════════════════════════════════════════════════════════════
    # Video loading / playback
    # ══════════════════════════════════════════════════════════════════════════

    def _load_video(self, pos: int):
        if pos >= len(self._video_list):
            messagebox.showinfo("Done", "All videos annotated!")
            return
        self._video_pos = pos
        v = self._current_video()

        self._stop_playback()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(str(v["path"]))
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open:\n{v['path']}")
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._vid_w       = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._vid_h       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cur_frame    = 0
        self.annotations  = []
        self._undo_stack  = []
        self._pending     = {}
        self._state       = self.IDLE
        self._sel_idx     = -1

        # check for existing partial save (resume within video)
        ann_path = v["path"].parent / "annotations.json"
        if ann_path.exists():
            try:
                with open(ann_path, encoding="utf-8") as f:
                    saved = json.load(f)
                self.annotations = saved.get("steps", [])
            except Exception:
                pass

        # reset audio state for new video
        self._audio_file  = None
        self._audio_ready = False
        if _AUDIO_OK:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        # start background audio extraction
        self._start_audio_prep(v["path"])

        self._update_top_bar()
        self._refresh_list()
        self._show_frame(0)
        self._update_state_label()
        self._btn_play.config(text="▶  Play")

    def _show_frame(self, idx: int):
        """Seek to a specific frame (used for step/jump/seek — not during playback)."""
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cur_frame = idx
        with self._cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._render_rgb(rgb)
        self._update_timeline()

    def _render_rgb(self, rgb: np.ndarray):
        """Letterbox-fit an RGB frame onto the canvas (called from main thread only)."""
        h, w = rgb.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        nw, nh = int(w * scale), int(h * scale)
        img = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        canvas_img = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
        ox = (CANVAS_W - nw) // 2
        oy = (CANVAS_H - nh) // 2
        canvas_img.paste(img, (ox, oy))

        self._disp_x     = ox
        self._disp_y     = oy
        self._disp_scale = scale
        self._vid_w      = w
        self._vid_h      = h

        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("frame")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="frame")
        self.canvas.tag_lower("frame")
        self._draw_annotation_overlays()

    def _draw_annotation_overlays(self):
        self.canvas.delete("ann_overlay")
        for i, ann in enumerate(self.annotations):
            atype = ann.get("action_type", "")
            clr   = ACTION_COLORS.get(atype, FG2)
            fi    = ann.get("frame_index", -1)
            if fi != self.cur_frame:
                continue
            self._draw_ann_on_canvas(ann, clr, tag="ann_overlay")

    def _draw_ann_on_canvas(self, ann: dict, color: str, tag: str = "tmp"):
        atype = ann.get("action_type", "")
        if atype in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            bbox = ann.get("bbox")
            if bbox:
                self._draw_norm_bbox(bbox, color, tag)
            else:
                pt = ann.get("point", [0.5, 0.5])
                cx, cy = self._norm_to_canvas(*pt)
                r = 6
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                        outline=color, width=2, tags=tag)
                self.canvas.create_line(cx-10, cy, cx+10, cy,
                                        fill=color, width=1, tags=tag)
                self.canvas.create_line(cx, cy-10, cx, cy+10,
                                        fill=color, width=1, tags=tag)
        elif atype == "SCROLL":
            bbox = ann.get("bbox")
            if bbox:
                self._draw_norm_bbox(bbox, color, tag)
            direction = ann.get("direction", "")
            pos = ann.get("position", [0.5, 0.5])
            cx, cy = self._norm_to_canvas(*pos)
            arrow_map = {"up": (0,-20), "down": (0,20),
                         "left": (-20,0), "right": (20,0)}
            dx, dy = arrow_map.get(direction, (0, 0))
            self.canvas.create_line(cx, cy, cx+dx, cy+dy,
                                    fill=color, width=2, arrow="last", tags=tag)
        elif atype == "DRAG":
            sb = ann.get("start_bbox")
            eb = ann.get("end_bbox")
            if sb:
                self._draw_norm_bbox(sb, color, tag)
            if eb:
                self._draw_norm_bbox(eb, ACC2, tag)
            s = ann.get("start", [0.3, 0.5])
            e = ann.get("end",   [0.7, 0.5])
            sx, sy = self._norm_to_canvas(*s)
            ex, ey = self._norm_to_canvas(*e)
            self.canvas.create_line(sx, sy, ex, ey,
                                    fill=color, width=2, arrow="last",
                                    dash=(4, 3), tags=tag)

    def _draw_norm_bbox(self, bbox_norm: list, color: str, tag: str):
        """bbox_norm = [x, y, w, h] normalised [0,1]"""
        x, y, w, h = bbox_norm
        x1, y1 = self._norm_to_canvas(x, y)
        x2, y2 = self._norm_to_canvas(x + w, y + h)
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     outline=color, width=2, tags=tag)

    # ── Audio ──────────────────────────────────────────────────────────────

    def _start_audio_prep(self, video_path: Path):
        """Kick off background extraction of audio to a cached WAV."""
        if not _AUDIO_OK or not FFMPEG:
            return
        cache = video_path.parent / "_audio_cache.wav"
        if cache.exists():
            self._audio_file  = str(cache)
            self._audio_ready = True
            return
        self._audio_ready = False

        def _extract():
            try:
                subprocess.run(
                    [FFMPEG, "-y", "-i", str(video_path),
                     "-vn", "-ar", "44100", "-ac", "2", "-f", "wav", str(cache)],
                    capture_output=True, timeout=120,
                )
                if cache.exists():
                    self._audio_file  = str(cache)
                    self._audio_ready = True
            except Exception:
                pass

        t = threading.Thread(target=_extract, daemon=True)
        t.start()
        self._audio_prep_thread = t

    def _audio_play(self):
        """Start audio from cur_frame position (call after video thread starts)."""
        if not _AUDIO_OK or not self._audio_ready or not self._audio_file:
            return
        if self._speed != 1.0:
            return   # skip audio at non-1x speeds (no pitch correction)
        try:
            pos = self.cur_frame / self.fps
            pygame.mixer.music.load(self._audio_file)
            pygame.mixer.music.play(start=pos)
        except Exception:
            pass

    def _audio_pause(self):
        if not _AUDIO_OK:
            return
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def _audio_stop(self):
        if not _AUDIO_OK:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    # ── Playback ───────────────────────────────────────────────────────────

    def _stop_playback(self):
        """Stop the decode thread and drain the queue."""
        if self.playing:
            self.playing = False
            self._btn_play.config(text="▶  Play")
        self._audio_pause()
        self._stop_event.set()
        # drain so the thread can exit if it's blocked on put()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if self._poll_job:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None

    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        else:
            self.playing = True
            self._btn_play.config(text="⏸  Pause")
            self._stop_event.clear()
            t = threading.Thread(target=self._decode_thread, daemon=True)
            t.start()
            self._audio_play()
            self._poll_queue()

    def _decode_thread(self):
        """Background thread: read frames and push (idx, rgb_array) to queue."""
        speed = self._speed
        interval = 1.0 / (self.fps * speed)
        start_frame = self.cur_frame

        with self._cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fi = start_frame
        # how many source frames to advance per display frame
        frame_step = max(1, round(speed))
        while not self._stop_event.is_set():
            t0 = time.monotonic()

            with self._cap_lock:
                # for speed > 1 seek ahead so we actually skip frames
                if frame_step > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, bgr = self.cap.read()

            if not ok or fi >= self.total_frames:
                try:
                    self._frame_queue.put(("eof", fi, None), timeout=0.5)
                except queue.Full:
                    pass
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                self._frame_queue.put(("frame", fi, rgb), timeout=0.5)
            except queue.Full:
                if self._stop_event.is_set():
                    break

            fi += frame_step
            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _poll_queue(self):
        """Main-thread poller: pull frames from queue and display them."""
        try:
            kind, fi, rgb = self._frame_queue.get_nowait()
        except queue.Empty:
            kind = None

        if kind == "frame":
            self.cur_frame = fi
            self._render_rgb(rgb)
            self._update_timeline()
        elif kind == "eof":
            self._stop_playback()
            return

        if self.playing:
            self._poll_job = self.root.after(16, self._poll_queue)

    def _step(self, delta: int):
        if self.playing:
            self._stop_playback()
        self._show_frame(self.cur_frame + delta)

    def _jump_pct(self, pct: float):
        if self.playing:
            self._stop_playback()
        delta = int(self.total_frames * pct / 100)
        self._show_frame(self.cur_frame + delta)

    def _seek_secs(self, secs: float):
        if self.playing:
            self._stop_playback()
        delta = int(secs * self.fps)
        self._show_frame(self.cur_frame + delta)

    def _set_speed(self, speed: float):
        was_playing = self.playing
        if was_playing:
            self._stop_playback()
        self._speed = speed
        for spd, btn in self._speed_btns.items():
            active = (spd == speed)
            btn.config(bg=ACC if active else BG3,
                       fg=BG if active else FG2)
        if was_playing:
            self._toggle_play()

    def _on_timeline_drag(self, val):
        if self.total_frames < 1 or self._updating_timeline:
            return
        if self.playing:
            self._stop_playback()
        frac = float(val) / 1000.0
        idx  = int(frac * (self.total_frames - 1))
        self._show_frame(idx)

    def _update_timeline(self):
        if self._updating_timeline:
            return
        self._updating_timeline = True
        try:
            if self.total_frames > 0:
                frac = self.cur_frame / (self.total_frames - 1)
                self._timeline.set(frac * 1000)
            ts_cur = self._frame_to_ts(self.cur_frame)
            ts_tot = self._frame_to_ts(self.total_frames - 1)
            self._lbl_time.config(text=f"{ts_cur} / {ts_tot}")
        finally:
            self._updating_timeline = False

    def _frame_to_ts(self, frame: int) -> str:
        secs = int(frame / self.fps)
        return f"{secs // 60}:{secs % 60:02d}"

    # ── Coordinate conversion ──────────────────────────────────────────────

    def _canvas_to_norm(self, cx: float, cy: float):
        nx = (cx - self._disp_x) / (self._vid_w * self._disp_scale)
        ny = (cy - self._disp_y) / (self._vid_h * self._disp_scale)
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _norm_to_canvas(self, nx: float, ny: float):
        cx = self._disp_x + nx * self._vid_w * self._disp_scale
        cy = self._disp_y + ny * self._vid_h * self._disp_scale
        return cx, cy

    def _bbox_from_canvas_coords(self, x0, y0, x1, y1) -> list:
        """Return [nx, ny, nw, nh] from two canvas corners (order-independent)."""
        cx0, cy0 = min(x0, x1), min(y0, y1)
        cx1, cy1 = max(x0, x1), max(y0, y1)
        nx0, ny0 = self._canvas_to_norm(cx0, cy0)
        nx1, ny1 = self._canvas_to_norm(cx1, cy1)
        return [nx0, ny0, nx1 - nx0, ny1 - ny0]

    # ══════════════════════════════════════════════════════════════════════════
    # Mouse (bbox drawing)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_canvas_press(self, event):
        if self._state not in (self.DRAW_BBOX, self.DRAW_BBOX2):
            return
        self._bbox_start_canvas = (event.x, event.y)
        if self._bbox_rect_id:
            self.canvas.delete(self._bbox_rect_id)
            self._bbox_rect_id = None

    def _on_canvas_drag(self, event):
        if self._state not in (self.DRAW_BBOX, self.DRAW_BBOX2):
            return
        if self._bbox_start_canvas is None:
            return
        x0, y0 = self._bbox_start_canvas
        if self._bbox_rect_id:
            self.canvas.delete(self._bbox_rect_id)
        color = ACC if self._state == self.DRAW_BBOX else ACC2
        self._bbox_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=color, width=2, dash=(4, 2), tags="tmp_bbox"
        )

    def _on_canvas_release(self, event):
        if self._state not in (self.DRAW_BBOX, self.DRAW_BBOX2):
            return
        if self._bbox_start_canvas is None:
            return
        x0, y0 = self._bbox_start_canvas
        x1, y1 = event.x, event.y
        self._bbox_start_canvas = None

        # require a minimum drag
        if abs(x1 - x0) < 5 and abs(y1 - y0) < 5:
            return

        bbox = self._bbox_from_canvas_coords(x0, y0, x1, y1)
        cx   = (bbox[0] + bbox[2] / 2)
        cy   = (bbox[1] + bbox[3] / 2)

        atype = self._pending.get("action_type", "")

        if self._state == self.DRAW_BBOX:
            if atype in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                self._pending["bbox"]  = bbox
                self._pending["point"] = [cx, cy]
                self._commit_pending()
            elif atype == "SCROLL":
                self._pending["bbox"]     = bbox
                self._pending["position"] = [cx, cy]
                self._commit_pending()
            elif atype == "DRAG":
                self._pending["start_bbox"] = bbox
                self._pending["start"]      = [cx, cy]
                # move to second bbox
                self._state = self.DRAW_BBOX2
                self._update_state_label()

        elif self._state == self.DRAW_BBOX2:
            # DRAG end bbox
            self._pending["end_bbox"] = bbox
            self._pending["end"]      = [cx, cy]
            self._commit_pending()

        if self._bbox_rect_id:
            self.canvas.delete(self._bbox_rect_id)
            self._bbox_rect_id = None

    # ══════════════════════════════════════════════════════════════════════════
    # Keyboard
    # ══════════════════════════════════════════════════════════════════════════

    def _on_key(self, event):
        sym   = event.keysym

        # --- SCROLL direction ---
        if self._state == self.WAIT_DIR:
            if sym in DIR_KEYS:
                self._pending["direction"] = DIR_KEYS[sym]
                # now draw bbox for scroll position
                self._state = self.DRAW_BBOX
                self._update_state_label()
            return

        # --- global shortcuts that don't interfere ---
        if sym in ("Control_L", "Control_R", "Shift_L", "Shift_R"):
            return

        # 1-8: select action type
        if sym in ACTION_LABELS:
            self._select_action(ACTION_LABELS[sym])
            return

    def _select_action(self, action: str):
        if self.playing:
            self._toggle_play()
        self._cancel_pending()
        self._pending = {
            "action_type":  action,
            "frame_index":  self.cur_frame,
            "timestamp":    self.cur_frame / self.fps,
        }

        # highlight button
        for lbl, btn in self._action_btns.items():
            btn.config(bg=BG3 if lbl != action else ACTION_COLORS[action],
                       fg=ACTION_COLORS[lbl] if lbl != action else BG)

        if action == "FINISH":
            self._commit_pending()
        elif action == "HOTKEY":
            key = simpledialog.askstring(
                "HOTKEY", "Type the hotkey combination:\n"
                          "e.g.  ctrl+s   ctrl+shift+z   enter   f2   escape",
                parent=self.root
            )
            if key and key.strip():
                self._pending["key"] = key.strip().lower()
                self._commit_pending()
            else:
                self._cancel_pending()
        elif action == "TYPE":
            text = simpledialog.askstring(
                "TYPE", "Enter the text to type:",
                parent=self.root
            )
            if text is not None:
                self._pending["text"] = text
                self._commit_pending()
            else:
                self._cancel_pending()
        elif action == "SCROLL":
            self._state = self.WAIT_DIR
        elif action == "DRAG":
            self._state = self.DRAW_BBOX   # first bbox
            self._bbox_stage = 1
        else:
            # CLICK / DOUBLE_CLICK / RIGHT_CLICK
            self._state = self.DRAW_BBOX

        self._update_state_label()

    # ══════════════════════════════════════════════════════════════════════════
    # Annotation management
    # ══════════════════════════════════════════════════════════════════════════

    def _commit_pending(self):
        ann = dict(self._pending)
        self._undo_stack.append(("add", len(self.annotations)))
        self.annotations.append(ann)
        self._pending = {}
        self._state   = self.IDLE

        # reset action button highlight
        for btn in self._action_btns.values():
            btn.config(bg=BG3)
        for lbl, btn in self._action_btns.items():
            btn.config(fg=ACTION_COLORS[lbl])

        self._refresh_list()
        self._draw_annotation_overlays()
        self._update_state_label()

    def _cancel_pending(self):
        self._pending = {}
        self._state   = self.IDLE
        if self._bbox_rect_id:
            self.canvas.delete(self._bbox_rect_id)
            self._bbox_rect_id = None
        for lbl, btn in self._action_btns.items():
            btn.config(bg=BG3, fg=ACTION_COLORS[lbl])
        self._update_state_label()

    def _undo(self):
        if not self._undo_stack:
            return
        op, idx = self._undo_stack.pop()
        if op == "add" and 0 <= idx < len(self.annotations):
            self.annotations.pop(idx)
        self._refresh_list()
        self._draw_annotation_overlays()
        self._update_state_label()

    def _delete_selected(self):
        sel = self._ann_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.annotations):
            self.annotations.pop(idx)
            self._refresh_list()
            self._draw_annotation_overlays()

    def _refresh_list(self):
        self._ann_list.delete(0, "end")
        for i, ann in enumerate(self.annotations):
            self._ann_list.insert("end", self._format_ann(i, ann))
        if self.annotations:
            self._ann_list.see("end")

    def _format_ann(self, i: int, ann: dict) -> str:
        atype = ann.get("action_type", "?")
        ts    = ann.get("timestamp", 0.0)
        fi    = ann.get("frame_index", 0)
        ts_s  = f"{int(ts)//60}:{int(ts)%60:02d}"
        base  = f"#{i+1:02d}  [{ts_s} f{fi}]  {atype}"
        if atype in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            pt = ann.get("point", [0, 0])
            base += f"  ({pt[0]:.3f},{pt[1]:.3f})"
        elif atype == "SCROLL":
            base += f"  {ann.get('direction','?')}"
        elif atype == "DRAG":
            s = ann.get("start", [0, 0])
            e = ann.get("end",   [0, 0])
            base += f"  ({s[0]:.2f},{s[1]:.2f})→({e[0]:.2f},{e[1]:.2f})"
        elif atype == "TYPE":
            t = ann.get("text", "")
            base += f'  "{t[:20]}"'
        elif atype == "HOTKEY":
            base += f"  {ann.get('key','?')}"
        return base

    def _on_list_select(self, event):
        sel = self._ann_list.curselection()
        if sel:
            self._sel_idx = sel[0]

    def _on_list_goto(self, event):
        sel = self._ann_list.curselection()
        if sel:
            idx = sel[0]
            if 0 <= idx < len(self.annotations):
                fi = self.annotations[idx].get("frame_index", 0)
                self._show_frame(fi)

    # ══════════════════════════════════════════════════════════════════════════
    # Save & next
    # ══════════════════════════════════════════════════════════════════════════

    def _save_and_next(self):
        if not self.annotations:
            if not messagebox.askyesno("Empty", "No annotations yet. Save empty and skip?"):
                return
        self._do_save()
        next_pos = self._find_next_undone(self._video_pos + 1)
        if next_pos is None:
            messagebox.showinfo("Done", "All videos have been annotated!")
            return
        self._load_video(next_pos)

    def _skip_video(self):
        next_pos = self._find_next_undone(self._video_pos + 1)
        if next_pos is None:
            messagebox.showinfo("Done", "No more unannotated videos.")
            return
        self._load_video(next_pos)

    def _go_prev(self):
        if self._video_pos <= 0:
            return
        self._nav_away()
        self._load_video(self._video_pos - 1)

    def _go_next(self):
        if self._video_pos >= len(self._video_list) - 1:
            return
        self._nav_away()
        self._load_video(self._video_pos + 1)

    def _nav_away(self):
        """Ask to save unsaved work before navigating away."""
        if self.annotations:
            if messagebox.askyesno("Save?", "Save current annotations before navigating?"):
                self._do_save()

    def _find_next_undone(self, start: int) -> Optional[int]:
        for i in range(start, len(self._video_list)):
            ann_path = self._video_list[i]["path"].parent / "annotations.json"
            if not ann_path.exists():
                return i
        return None

    def _do_save(self):
        v        = self._current_video()
        vid_dir  = v["path"].parent

        # ── Extract frames ─────────────────────────────────────────────────
        frames_dir = vid_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        extracted = {}  # frame_index → filename
        cap2 = cv2.VideoCapture(str(v["path"]))
        for ann in self.annotations:
            fi = ann.get("frame_index", 0)
            if fi in extracted:
                continue
            cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap2.read()
            if ok:
                fname = f"frame_{fi:06d}.jpg"
                cv2.imwrite(str(frames_dir / fname), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                extracted[fi] = fname
        cap2.release()

        # ── Build step list ────────────────────────────────────────────────
        steps = []
        for ann in self.annotations:
            fi    = ann.get("frame_index", 0)
            fname = extracted.get(fi, "")
            step  = {k: v2 for k, v2 in ann.items()}
            step["frame_file"] = f"frames/{fname}" if fname else ""
            steps.append(step)

        payload = {
            "video_id":    v["id"],
            "task":        v["task"],
            "title":       v["title"],
            "video_path":  str(v["path"].relative_to(VIDEO_ROOT)),
            "steps":       steps,
        }
        ann_path = vid_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved: {ann_path}")

        self._update_top_bar()

    # ══════════════════════════════════════════════════════════════════════════
    # UI helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _update_top_bar(self):
        v     = self._current_video()
        total = len(self._video_list)
        done  = sum(1 for vv in self._video_list
                    if (vv["path"].parent / "annotations.json").exists())
        already_done = (v["path"].parent / "annotations.json").exists()
        marker = "  [annotated]" if already_done else ""
        self._lbl_task.config(
            text=f"[{v['domain']}]  {v['task']}{marker}",
            fg=WARN if already_done else FG,
        )
        self._lbl_progress.config(
            text=f"{done}/{total} done  •  #{self._video_pos + 1}/{total}"
        )

    STATE_HINTS = {
        IDLE:        ("Select an action type  (1-8)",
                      "Space=play  ←/→=frame  ↑/↓=5%\nEnter=save & next   Ctrl+Z=undo"),
        WAIT_DIR:    ("SCROLL — press arrow key for direction",
                      "↑ up   ↓ down   ← left   → right\nEsc=cancel"),
        DRAW_BBOX:   ("Draw bbox on canvas  (click & drag)",
                      "Drag to mark the target region\nEsc=cancel"),
        DRAW_BBOX2:  ("DRAG — draw END bbox",
                      "Drag to mark the drag destination\nEsc=cancel"),
    }

    def _update_state_label(self):
        # pending action type
        atype = self._pending.get("action_type", "")
        if atype:
            clr = ACTION_COLORS.get(atype, ACC)
        else:
            clr = ACC

        title, hint = self.STATE_HINTS.get(
            self._state,
            (f"State: {self._state}", "")
        )
        if atype and self._state != self.IDLE:
            title = f"[{atype}]  {title}"
        self._lbl_state.config(text=title, fg=clr)
        self._lbl_hint.config(text=hint)

    def _on_close(self):
        if self.annotations:
            if messagebox.askyesno("Quit", "Save current video before quitting?"):
                self._do_save()
        self._stop_playback()
        self._audio_stop()
        with self._cap_lock:
            if self.cap:
                self.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════

def _apply_dark_ttk_theme(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Horizontal.TScale",
                    background=BG2, troughcolor=SCROLL_CLR,
                    sliderrelief="flat", sliderlength=14)
    style.configure("Vertical.TScrollbar",
                    background=BG3, troughcolor=BG2,
                    arrowcolor=FG2, bordercolor=BG2)
    style.configure("TSeparator", background=BG3)


def main():
    root = tk.Tk()
    root.geometry(f"{WIN_W}x{WIN_H}")
    _apply_dark_ttk_theme(root)
    app = AnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
