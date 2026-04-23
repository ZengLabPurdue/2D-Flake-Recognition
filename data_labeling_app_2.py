"""
Cross-platform flake labeling — Sobel on images under a ``Raw/`` folder, four classes:
Good / Bad / Unsure / No Flake. Each crop is saved under ``Labeled Data/<Class>/``.
Labels persist per-image across navigation. Review panel is collapsible.
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
import sys
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

home_dir = Path(os.path.dirname(os.path.abspath(__file__)))
flake_finder_dir = home_dir / "Flake Recognition"
sys.path.insert(0, str(flake_finder_dir))

import flake_finder  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CROP_SCALE = 1.2

# (slug, button label, outline color key)
CLASS_DEFS: list[tuple[str, str, str]] = [
    ("good",     "Good",     "good"),
    ("bad",      "Bad",      "bad"),
    ("unsure",   "Unsure",   "unsure"),
    ("no_flake", "No Flake", "no_flake"),
]

CLASS_SAVE_DIRS: dict[str, str] = {
    "good":     "Good",
    "bad":      "Bad",
    "unsure":   "Unsure",
    "no_flake": "No_Flake",
}

OUTLINE_HEX = {
    "good":     "#3fb950",
    "bad":      "#f85149",
    "unsure":   "#d29922",
    "no_flake": "#8b949e",
}

# Dark tint of each class color — used for armed button background
CLASS_ARMED_BG = {
    "good":     "#0d2117",
    "bad":      "#2b0f0d",
    "unsure":   "#2a1c07",
    "no_flake": "#1c212b",
}

# ttk style id per slug ("no_flake" → "NoFlake")
_STYLE_ID = {s: "".join(w.capitalize() for w in s.split("_")) for s, _, _ in CLASS_DEFS}

SEGMENTATION_LINE_HEX = "#ffffff"
CONTOUR_WIDTH         = 1
CONTOUR_WIDTH_HOVER   = 2
BBOX_WIDTH            = 2
BBOX_WIDTH_HOVER      = 3
BOX_SELECT_MIN_DRAG   = 5

# ── Dark theme palette ────────────────────────────────────────────────────────
T_BG      = "#111318"   # root background / canvas well
T_PANEL   = "#17191e"   # left sidebar
T_CARD    = "#1f2229"   # labelframe / card
T_SURFACE = "#282d38"   # interactive elements (buttons, scales, inputs)
T_BORDER  = "#353c49"   # borders / separators
T_FG      = "#dce1f0"   # primary text
T_MUTED   = "#7681a0"   # secondary / muted text
T_DIM     = "#464f6b"   # disabled / very dim
T_ACCENT  = "#5b8af7"   # blue accent
T_CANVAS  = "#090c10"   # main labeling canvas background


def _apply_dark_style(root: tk.Tk) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # Global defaults
    style.configure(".",
        background=T_PANEL, foreground=T_FG,
        fieldbackground=T_SURFACE, bordercolor=T_BORDER,
        troughcolor=T_SURFACE, selectbackground=T_ACCENT,
        selectforeground="#fff", font=("TkDefaultFont", 10),
    )

    # Frames
    style.configure("TFrame",         background=T_PANEL)
    style.configure("App.TFrame",     background=T_BG)
    style.configure("Panel.TFrame",   background=T_PANEL)
    style.configure("Card.TFrame",    background=T_CARD)
    style.configure("Well.TFrame",    background=T_CANVAS)
    style.configure("Review.TFrame",  background=T_CARD)

    # Labels
    style.configure("TLabel",              background=T_PANEL,  foreground=T_FG)
    style.configure("Sidebar.TLabel",      background=T_PANEL,  foreground=T_FG)
    style.configure("Card.TLabel",         background=T_CARD,   foreground=T_FG)
    style.configure("Muted.TLabel",        background=T_PANEL,  foreground=T_MUTED)
    style.configure("CardMuted.TLabel",    background=T_CARD,   foreground=T_MUTED)
    style.configure("Well.TLabel",         background=T_CANVAS, foreground=T_MUTED,
                    font=("TkDefaultFont", 13))
    style.configure("Status.TLabel",       background=T_CARD,   foreground=T_MUTED,
                    font=("TkDefaultFont", 9), padding=(10, 5))
    style.configure("ImageName.TLabel",    background=T_CARD,   foreground=T_FG,
                    font=("TkDefaultFont", 9, "bold"))
    style.configure("Counter.TLabel",      background=T_CARD,   foreground=T_MUTED,
                    font=("TkDefaultFont", 9))
    style.configure("SectionTitle.TLabel", background=T_PANEL,  foreground=T_MUTED,
                    font=("TkDefaultFont", 8, "bold"))

    # LabelFrames
    style.configure("Sidebar.TLabelframe",
        background=T_CARD, bordercolor=T_BORDER, relief="flat", padding=0)
    style.configure("Sidebar.TLabelframe.Label",
        background=T_CARD, foreground=T_MUTED, font=("TkDefaultFont", 8, "bold"),
        padding=(4, 0))

    # Separator
    style.configure("TSeparator", background=T_BORDER)

    # Buttons — base
    style.configure("TButton",
        background=T_SURFACE, foreground=T_FG, bordercolor=T_BORDER,
        padding=(8, 6), relief="flat", focusthickness=0)
    style.map("TButton",
        background=[("active", T_BORDER), ("pressed", T_DIM)],
        foreground=[("disabled", T_DIM)])

    style.configure("Nav.TButton",    padding=(8, 5), font=("TkDefaultFont", 9))
    style.configure("Accent.TButton", background=T_ACCENT, foreground="#fff",
                    padding=(8, 6), font=("TkDefaultFont", 9, "bold"))
    style.map("Accent.TButton",
        background=[("active", "#7aa4fa"), ("pressed", "#3f6bde")])

    # Review toggle header button
    style.configure("ReviewToggle.TButton",
        background=T_CARD, foreground=T_MUTED,
        padding=(10, 6), font=("TkDefaultFont", 9, "bold"), relief="flat",
        bordercolor=T_BORDER)
    style.map("ReviewToggle.TButton",
        background=[("active", T_SURFACE)],
        foreground=[("active", T_FG)])

    # Thin dark scrollbar for the sidebar
    style.configure("Sidebar.Vertical.TScrollbar",
        background=T_SURFACE, troughcolor=T_PANEL,
        bordercolor=T_PANEL, arrowcolor=T_BORDER,
        relief="flat", width=6)
    style.map("Sidebar.Vertical.TScrollbar",
        background=[("active", T_BORDER)])

    # Class buttons — one normal + one armed variant per class
    style.configure("Class.TButton", padding=(10, 9), font=("TkDefaultFont", 10, "bold"),
                    relief="flat")
    for slug, _, _ in CLASS_DEFS:
        sid   = _STYLE_ID[slug]
        color = OUTLINE_HEX[slug]
        abg   = CLASS_ARMED_BG[slug]
        style.configure(f"{sid}.Class.TButton",
            background=T_CARD, foreground=color, bordercolor=T_BORDER)
        style.map(f"{sid}.Class.TButton",
            background=[("active", T_SURFACE)],
            foreground=[("active", color)])
        style.configure(f"{sid}Armed.Class.TButton",
            background=abg, foreground=color, bordercolor=color)
        style.map(f"{sid}Armed.Class.TButton",
            background=[("active", abg)],
            foreground=[("active", color)])

    # Combobox
    style.configure("TCombobox",
        fieldbackground=T_SURFACE, background=T_SURFACE,
        foreground=T_FG, bordercolor=T_BORDER, arrowcolor=T_FG,
        selectbackground=T_SURFACE, selectforeground=T_FG)
    style.map("TCombobox",
        fieldbackground=[("readonly", T_SURFACE)],
        foreground=[("readonly", T_FG)],
        selectbackground=[("readonly", T_SURFACE)])

    return style


# ── Tiny helper ───────────────────────────────────────────────────────────────

def _hex_label(parent, master_bg, text="", var=None, **kw) -> tk.Label:
    """A tk.Label with explicit bg matching the dark theme frame."""
    kwargs = dict(bg=master_bg, fg=T_FG, anchor="w")
    kwargs.update(kw)
    if var is not None:
        return tk.Label(parent, textvariable=var, **kwargs)
    return tk.Label(parent, text=text, **kwargs)


# ── Main application ──────────────────────────────────────────────────────────

class DataLabelingApp2:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Flake Labeler")
        self.root.minsize(980, 660)
        self.root.geometry("1260x820")
        self.root.configure(bg=T_BG)

        self._style = _apply_dark_style(root)

        self.save_folder_path = home_dir / "Labeled Data"
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self._ensure_class_dirs()

        self.status_var = tk.StringVar(value=f"Save folder: {self.save_folder_path}")

        # Image / contour state
        self.images: list[Path | str] = []
        self._folder_for_image: list[Path | None] = []
        self.current_image: Image.Image | None = None
        self.image_index = -1
        self.contours: list = []
        self.contour_labels: list[str | None] = []
        self.hover_idx: int | None = None
        self.active_label: str | None = None
        self.contour_saved_paths: dict[int, Path] = {}

        # Label persistence: image path str → (contour_labels list, saved_paths dict)
        self._image_label_cache: dict[str, tuple[list, dict]] = {}

        # Review state
        self.review_paths: list[Path] = []
        self.review_index = 0
        self._review_tk_img: ImageTk.PhotoImage | None = None
        self._review_win: tk.Toplevel | None = None   # popup window (created once)

        # Display geometry
        self._disp_w = 0
        self._disp_h = 0
        self._off_x  = 0
        self._off_y  = 0
        self._img_w  = 0
        self._img_h  = 0
        self._tk_image: ImageTk.PhotoImage | None = None

        # Box-select drag state
        self._box_dragging        = False
        self._box_drag_start: tuple[float, float] | None = None
        self._box_drag_current: tuple[float, float] | None = None
        self._box_pending_rect: tuple[float, float, float, float] | None = None
        self._box_pending_indices: list[int] = []

        self._build_layout()
        self._bind_keys()
        self.create_menu()

        self.label_area.bind("<Configure>", self.on_resize)
        self._review_refresh_list()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="App.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        # Left sidebar — scrollable via canvas+frame
        _sidebar_outer = tk.Frame(body, bg=T_PANEL, width=308)
        _sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        _sidebar_outer.pack_propagate(False)

        _sidebar_scroll = ttk.Scrollbar(
            _sidebar_outer, orient=tk.VERTICAL,
            style="Sidebar.Vertical.TScrollbar",
        )
        _sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._sidebar_canvas = tk.Canvas(
            _sidebar_outer, bg=T_PANEL,
            highlightthickness=0, bd=0,
            yscrollcommand=_sidebar_scroll.set,
        )
        self._sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _sidebar_scroll.config(command=self._sidebar_canvas.yview)

        self.left_panel = tk.Frame(self._sidebar_canvas, bg=T_PANEL)
        _win_id = self._sidebar_canvas.create_window(
            (0, 0), window=self.left_panel, anchor="nw",
        )

        def _on_inner_configure(_e):
            self._sidebar_canvas.configure(
                scrollregion=self._sidebar_canvas.bbox("all")
            )

        def _on_canvas_resize(e):
            self._sidebar_canvas.itemconfig(_win_id, width=e.width)

        self.left_panel.bind("<Configure>", _on_inner_configure)
        self._sidebar_canvas.bind("<Configure>", _on_canvas_resize)

        # Activate mousewheel only when cursor is over the sidebar
        def _mw(e):
            self._sidebar_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")

        self._sidebar_canvas.bind(
            "<Enter>", lambda _e: self._sidebar_canvas.bind_all("<MouseWheel>", _mw)
        )
        self._sidebar_canvas.bind(
            "<Leave>", lambda _e: self._sidebar_canvas.unbind_all("<MouseWheel>")
        )

        # Right area
        right_area = ttk.Frame(body, style="Well.TFrame")
        right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas container (fills all available space)
        self.label_area = ttk.Frame(right_area, style="Well.TFrame")
        self.label_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Placeholder text (shown when no image loaded)
        self.right_placeholder = ttk.Label(
            self.label_area,
            text=(
                "Open a raw image or folder\n"
                "File → Open Raw Image  /  Open Folder\n\n"
                "Files must be inside a folder named Raw/\n"
                "Arm a class (1–4), then click or drag to label"
            ),
            style="Well.TLabel",
            justify=tk.CENTER,
        )
        self.right_placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Main canvas
        self.canvas = tk.Canvas(self.label_area, bg=T_CANVAS, highlightthickness=0, bd=0)
        self.canvas.bind("<Configure>",      self._on_canvas_configure)
        self.canvas.bind("<Motion>",         self._on_canvas_motion)
        self.canvas.bind("<ButtonPress-1>",  self._on_canvas_button_press)
        self.canvas.bind("<B1-Motion>",      self._on_canvas_b1_motion)
        self.canvas.bind("<ButtonRelease-1>",self._on_canvas_button_release)
        self.canvas.bind("<Button-3>",       self._on_canvas_right_click)
        self.canvas.bind("<Leave>",          self._on_canvas_leave)

        # Build sidebar after so self.canvas exists
        self._build_sidebar()

        # Status bar
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.BOTTOM, fill=tk.X
        )

    def _build_sidebar(self):
        PAD = 10

        def section(title: str) -> tk.Frame:
            """Returns an inner content Frame inside a titled card."""
            lf = ttk.LabelFrame(self.left_panel, text=title.upper(),
                                style="Sidebar.TLabelframe", padding=(10, 8))
            lf.pack(fill=tk.X, padx=PAD, pady=(0, 8))
            lf.columnconfigure(0, weight=1)
            inner = tk.Frame(lf, bg=T_CARD)
            inner.pack(fill=tk.X)
            return inner

        tk.Frame(self.left_panel, bg=T_PANEL, height=10).pack()  # top spacer

        # ── Current image ──────────────────────────────────────────────────
        img_inner = section("Current Image")
        self.image_name_var    = tk.StringVar(value="No image loaded")
        self.image_counter_var = tk.StringVar(value="—")
        tk.Label(img_inner, textvariable=self.image_name_var,
                 bg=T_CARD, fg=T_FG, wraplength=252, justify="center",
                 font=("TkDefaultFont", 9, "bold")).pack(fill=tk.X, pady=(0, 4))
        tk.Label(img_inner, textvariable=self.image_counter_var,
                 bg=T_CARD, fg=T_MUTED, font=("TkDefaultFont", 9)).pack(fill=tk.X)

        # ── Navigation ────────────────────────────────────────────────────
        nav_lf = ttk.LabelFrame(self.left_panel, text="NAVIGATE",
                                style="Sidebar.TLabelframe", padding=(10, 8))
        nav_lf.pack(fill=tk.X, padx=PAD, pady=(0, 8))
        nav_inner = tk.Frame(nav_lf, bg=T_CARD)
        nav_inner.pack(fill=tk.X)
        nav_inner.columnconfigure(0, weight=1)
        nav_inner.columnconfigure(1, weight=1)

        self.previous_image_button = ttk.Button(nav_inner, text="◀  Prev",
                                                style="Nav.TButton",
                                                command=self.previous_image)
        self.next_image_button = ttk.Button(nav_inner, text="Next  ▶",
                                            style="Nav.TButton",
                                            command=self.next_image)
        self.previous_image_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.next_image_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.previous_image_button.state(["disabled"])
        self.next_image_button.state(["disabled"])

        # ── Contour stats ─────────────────────────────────────────────────
        stats_inner = section("Contours")
        self.contour_count_var = tk.StringVar(value="Found: —")
        self.labeled_count_var = tk.StringVar(value="Labeled: —")
        self.hover_info_var    = tk.StringVar(value="Hover: —")
        self.active_label_var  = tk.StringVar(value="Armed: (none)")

        for var, color, bold in [
            (self.contour_count_var, T_FG,    False),
            (self.labeled_count_var, T_FG,    False),
            (self.hover_info_var,    T_MUTED, False),
            (self.active_label_var,  T_ACCENT, True),
        ]:
            font = ("TkDefaultFont", 9, "bold") if bold else ("TkDefaultFont", 9)
            tk.Label(stats_inner, textvariable=var, bg=T_CARD, fg=color,
                     anchor="w", font=font).pack(fill=tk.X, pady=1)

        # ── Classes ───────────────────────────────────────────────────────
        cls_lf = ttk.LabelFrame(self.left_panel, text="CLASSES  (keys 1–4)",
                                style="Sidebar.TLabelframe", padding=(10, 8))
        cls_lf.pack(fill=tk.X, padx=PAD, pady=(0, 8))
        cls_inner = tk.Frame(cls_lf, bg=T_CARD)
        cls_inner.pack(fill=tk.X)
        cls_inner.columnconfigure(0, weight=1)

        self.buttons: dict[str, ttk.Button] = {}
        for i, (slug, title, _) in enumerate(CLASS_DEFS):
            key = str(i + 1)
            sid = _STYLE_ID[slug]
            btn = ttk.Button(
                cls_inner,
                text=f"{key}   {title}",
                style=f"{sid}.Class.TButton",
                command=lambda s=slug: self.set_active_class(s),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=3)
            self.buttons[slug] = btn
            self.root.bind(f"<KeyPress-{key}>", lambda e, s=slug: self.set_active_class(s))

        # ── Sobel overlay ─────────────────────────────────────────────────
        sobel_inner = section("Sobel / Overlay")
        cb_kw = dict(bg=T_CARD, fg=T_FG, activebackground=T_CARD,
                     activeforeground=T_FG, selectcolor=T_SURFACE,
                     highlightthickness=0, bd=0, anchor="w",
                     font=("TkDefaultFont", 9))

        self.hover_only_var = tk.IntVar(value=1)
        tk.Checkbutton(sobel_inner, text="Show Sobel outline under mouse only",
                       variable=self.hover_only_var, onvalue=1, offvalue=0,
                       command=self.redraw_canvas, **cb_kw).pack(fill=tk.X)

        self.box_select_var = tk.IntVar(value=0)
        tk.Checkbutton(sobel_inner, text="Drag rectangle to label many",
                       variable=self.box_select_var, onvalue=1, offvalue=0,
                       command=self.redraw_canvas, **cb_kw).pack(fill=tk.X, pady=(6, 0))

        # Opacity slider
        op_row = tk.Frame(sobel_inner, bg=T_CARD)
        op_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(op_row, text="Overlay opacity:", bg=T_CARD, fg=T_MUTED,
                 font=("TkDefaultFont", 8), anchor="w").pack(side=tk.LEFT)
        self.overlay_opacity_var = tk.DoubleVar(value=0.35)
        _op_val_lbl = tk.Label(op_row, text="0.35", bg=T_CARD, fg=T_FG,
                               font=("TkDefaultFont", 8, "bold"), width=4, anchor="e")
        _op_val_lbl.pack(side=tk.RIGHT)
        def _op_cmd(v):
            _op_val_lbl.config(text=f"{float(v):.2f}")
            self.redraw_canvas()
        tk.Scale(op_row, variable=self.overlay_opacity_var, from_=0.0, to=1.0,
                 resolution=0.01, orient=tk.HORIZONTAL, bg=T_CARD, fg=T_MUTED,
                 troughcolor=T_SURFACE, highlightthickness=0, showvalue=False,
                 command=_op_cmd).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Frame(sobel_inner, bg=T_BORDER, height=1).pack(fill=tk.X, pady=(8, 4))

        self.suppress_yellow_var = tk.IntVar(value=0)
        tk.Checkbutton(sobel_inner, text="Suppress yellow edges",
                       variable=self.suppress_yellow_var, onvalue=1, offvalue=0,
                       command=self._reload_contours, **cb_kw).pack(fill=tk.X)

        def _yscale(parent, label, var, lo, hi):
            row = tk.Frame(parent, bg=T_CARD)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg=T_CARD, fg=T_MUTED,
                     font=("TkDefaultFont", 8), width=16, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text=str(var.get()), bg=T_CARD, fg=T_FG,
                               font=("TkDefaultFont", 8, "bold"), width=4, anchor="e")
            val_lbl.pack(side=tk.RIGHT)
            def _cmd(v, lbl=val_lbl):
                lbl.config(text=str(int(float(v))))
                self._reload_contours()
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=1,
                     orient=tk.HORIZONTAL, bg=T_CARD, fg=T_MUTED,
                     troughcolor=T_SURFACE, highlightthickness=0, showvalue=False,
                     command=_cmd).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.yellow_hue_lo_var  = tk.IntVar(value=18)
        self.yellow_hue_hi_var  = tk.IntVar(value=90)
        self.yellow_sat_min_var = tk.IntVar(value=40)
        self.yellow_val_min_var = tk.IntVar(value=154)
        _yscale(sobel_inner, "Hue lo (0–90):",   self.yellow_hue_lo_var,  0,  90)
        _yscale(sobel_inner, "Hue hi (0–90):",   self.yellow_hue_hi_var,  0,  90)
        _yscale(sobel_inner, "Sat min (0–255):", self.yellow_sat_min_var, 0, 255)
        _yscale(sobel_inner, "Val min (0–255):", self.yellow_val_min_var, 0, 255)

        tk.Frame(sobel_inner, bg=T_BORDER, height=1).pack(fill=tk.X, pady=(8, 4))

        self.filter_yellow_interior_var = tk.IntVar(value=0)
        tk.Checkbutton(sobel_inner, text="Drop yellow-interior contours",
                       variable=self.filter_yellow_interior_var, onvalue=1, offvalue=0,
                       command=self._reload_contours, **cb_kw).pack(fill=tk.X)

        self.yellow_interior_pct_var = tk.IntVar(value=50)
        _yscale(sobel_inner, "Interior % (0–100):", self.yellow_interior_pct_var, 1, 100)

        # ── Visibility ────────────────────────────────────────────────────
        vis_inner = section("Visibility")
        self.show_unlabeled_var = tk.IntVar(value=1)
        tk.Checkbutton(vis_inner, text="Unlabeled (white mask)",
                       variable=self.show_unlabeled_var, onvalue=1, offvalue=0,
                       command=self.redraw_canvas, **cb_kw).pack(fill=tk.X)

        self.filter_vars: dict[str, tk.IntVar] = {}
        for _slug, name, ok in CLASS_DEFS:
            v = tk.IntVar(value=1)
            self.filter_vars[ok] = v
            color = OUTLINE_HEX[ok]
            tk.Checkbutton(vis_inner, text=f"{name} (box)",
                           variable=v, onvalue=1, offvalue=0,
                           command=self.redraw_canvas,
                           fg=color, activeforeground=color,
                           **{k: cb_kw[k] for k in cb_kw if k not in ("fg", "activeforeground")}
                           ).pack(fill=tk.X)

        # ── Review window button ──────────────────────────────────────────
        self._review_btn = ttk.Button(
            self.left_panel,
            text="⊞   Review Saved Crops",
            style="ReviewToggle.TButton",
            command=self._open_review_window,
        )
        self._review_btn.pack(fill=tk.X, padx=PAD, pady=(0, 10))

        # ── Hints ─────────────────────────────────────────────────────────
        tk.Frame(self.left_panel, bg=T_PANEL, height=4).pack()
        tk.Label(self.left_panel,
                 text=(
                     "Right-click a flake to clear its label.\n"
                     "Up / Down arrows: navigate images.\n"
                     "Enter: confirm box selection.  Esc: cancel."
                 ),
                 bg=T_PANEL, fg=T_DIM, font=("TkDefaultFont", 8),
                 wraplength=272, justify="center").pack(padx=PAD, pady=(0, 10))

    # ── Review popup window ───────────────────────────────────────────────────

    def _open_review_window(self) -> None:
        """Create (first time) or raise the review popup window."""
        if self._review_win is not None and self._review_win.winfo_exists():
            self._review_win.deiconify()
            self._review_win.lift()
            self._review_show_current()
            return

        win = tk.Toplevel(self.root)
        win.title("Review Saved Crops")
        win.geometry("920x700")
        win.minsize(640, 480)
        win.configure(bg=T_BG)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)   # hide, don't destroy
        self._review_win = win

        # ── Header bar ────────────────────────────────────────────────────
        header = tk.Frame(win, bg=T_CARD, pady=0)
        header.pack(fill=tk.X)

        title_row = tk.Frame(header, bg=T_CARD)
        title_row.pack(fill=tk.X, padx=16, pady=10)

        tk.Label(title_row, text="REVIEW SAVED CROPS", bg=T_CARD, fg=T_MUTED,
                 font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)

        self._review_total_var = tk.StringVar(value="")
        tk.Label(title_row, textvariable=self._review_total_var, bg=T_CARD, fg=T_DIM,
                 font=("TkDefaultFont", 9)).pack(side=tk.RIGHT)

        tk.Frame(win, bg=T_BORDER, height=1).pack(fill=tk.X)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = tk.Frame(win, bg=T_PANEL, pady=0)
        toolbar.pack(fill=tk.X)

        tb_inner = tk.Frame(toolbar, bg=T_PANEL)
        tb_inner.pack(fill=tk.X, padx=14, pady=8)

        tk.Label(tb_inner, text="Folder:", bg=T_PANEL, fg=T_MUTED,
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 6))

        self.review_folder_var = tk.StringVar(value=CLASS_SAVE_DIRS["good"])
        self.review_folder_combo = ttk.Combobox(
            tb_inner, textvariable=self.review_folder_var,
            values=list(CLASS_SAVE_DIRS.values()),
            state="readonly", width=12,
        )
        self.review_folder_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.review_folder_combo.bind("<<ComboboxSelected>>",
                                      lambda _e: self._review_on_folder_change())

        ttk.Button(tb_inner, text="◀  Prev", style="Nav.TButton",
                   command=self._review_prev).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tb_inner, text="Next  ▶", style="Nav.TButton",
                   command=self._review_next).pack(side=tk.LEFT, padx=(0, 14))

        self.review_idx_var = tk.StringVar(value="0 / 0")
        tk.Label(tb_inner, textvariable=self.review_idx_var, bg=T_PANEL, fg=T_MUTED,
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT)

        tk.Frame(win, bg=T_BORDER, height=1).pack(fill=tk.X)

        # ── Crop preview canvas ───────────────────────────────────────────
        self.review_canvas = tk.Canvas(win, bg=T_CANVAS, highlightthickness=0, bd=0)
        self.review_canvas.pack(fill=tk.BOTH, expand=True)
        self.review_canvas.bind("<Configure>", self._on_review_canvas_configure)

        tk.Frame(win, bg=T_BORDER, height=1).pack(fill=tk.X)

        # ── Footer: filename + move-to ────────────────────────────────────
        footer = tk.Frame(win, bg=T_CARD)
        footer.pack(fill=tk.X)

        self.review_path_var = tk.StringVar(value="—")
        tk.Label(footer, textvariable=self.review_path_var,
                 bg=T_CARD, fg=T_MUTED, font=("TkDefaultFont", 9),
                 anchor="w", padx=14, pady=6).pack(fill=tk.X)

        tk.Frame(footer, bg=T_BORDER, height=1).pack(fill=tk.X)

        move_row = tk.Frame(footer, bg=T_CARD)
        move_row.pack(fill=tk.X, padx=14, pady=10)

        tk.Label(move_row, text="Move to:", bg=T_CARD, fg=T_MUTED,
                 font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 10))

        for slug, title, _ in CLASS_DEFS:
            color  = OUTLINE_HEX[slug]
            abg    = CLASS_ARMED_BG[slug]
            sid    = _STYLE_ID[slug]
            ttk.Button(move_row, text=title,
                       style=f"{sid}.Class.TButton",
                       command=lambda s=slug: self._review_move_to_slug(s),
                       ).pack(side=tk.LEFT, padx=4)

        # ── Window keyboard shortcuts ──────────────────────────────────────
        win.bind("<Left>",   lambda _e: self._review_prev())
        win.bind("<Right>",  lambda _e: self._review_next())
        win.bind("<Escape>", lambda _e: win.withdraw())
        for i, (slug, _, _) in enumerate(CLASS_DEFS):
            win.bind(str(i + 1), lambda _e, s=slug: self._review_move_to_slug(s))

        self._review_update_count_badge()
        self._review_refresh_list()

    def _review_update_count_badge(self) -> None:
        """Update crop count shown in the sidebar button and popup header."""
        total = 0
        for dname in CLASS_SAVE_DIRS.values():
            d = Path(self.save_folder_path) / dname
            if d.is_dir():
                total += sum(1 for p in d.iterdir()
                             if p.is_file() and p.suffix.lower() in {
                                 ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        # Update sidebar button text
        if hasattr(self, "_review_btn"):
            self._review_btn.config(
                text=f"⊞   Review Saved Crops  ({total})"
            )
        # Update popup header if open
        if hasattr(self, "_review_total_var"):
            self._review_total_var.set(f"{total} crops")

    # ── Key bindings ──────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<Up>",       self.previous_image)
        self.root.bind("<Down>",     self.next_image)
        self.root.bind("<Return>",   self._confirm_box_selection)
        self.root.bind("<KP_Enter>", self._confirm_box_selection)
        self.root.bind("<Escape>",   self._cancel_box_selection)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_under_raw_folder(path: Path) -> bool:
        try:
            return any(part.lower() == "raw" for part in Path(path).resolve().parts)
        except OSError:
            return False

    def _pil_for_labeling(self, opened: Path) -> Image.Image:
        return Image.open(Path(opened)).convert("RGB")

    def _outline_for_slug(self, slug: str | None) -> str | None:
        if slug is None:
            return None
        for s, _t, ol in CLASS_DEFS:
            if s == slug:
                return ol
        return None

    def _class_title(self, slug: str) -> str:
        for s, t, _ in CLASS_DEFS:
            if s == slug:
                return t
        return slug

    def _ensure_class_dirs(self) -> None:
        for dirname in CLASS_SAVE_DIRS.values():
            (Path(self.save_folder_path) / dirname).mkdir(parents=True, exist_ok=True)

    def _safe_stem(self) -> str:
        stem = Path(str(self.images[self.image_index])).stem
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)

    def _crop_basename(self, idx: int) -> str:
        return f"{self._safe_stem()}_c{idx}.png"

    # ── Label persistence ─────────────────────────────────────────────────────

    def _save_labels_for_current(self) -> None:
        """Snapshot the current image's labels into the cache."""
        if self.image_index < 0 or self.image_index >= len(self.images):
            return
        key = str(self.images[self.image_index])
        self._image_label_cache[key] = (
            list(self.contour_labels),
            dict(self.contour_saved_paths),
        )

    def _restore_labels_for_current(self) -> None:
        """Restore labels from cache (if available) for the just-loaded image."""
        if self.image_index < 0 or self.image_index >= len(self.images):
            return
        key = str(self.images[self.image_index])
        if key not in self._image_label_cache:
            return
        cached_labels, cached_paths = self._image_label_cache[key]
        n = len(self.contours)
        self.contour_labels = [
            cached_labels[i] if i < len(cached_labels) else None
            for i in range(n)
        ]
        self.contour_saved_paths = {
            i: cached_paths[i] for i in range(n) if i in cached_paths
        }

    # ── Crop saving ───────────────────────────────────────────────────────────

    def _save_crop_for_contour(self, idx: int, slug: str, *, skip_review_refresh: bool = False) -> None:
        if self.current_image is None or not self.contours or idx < 0 or idx >= len(self.contours):
            return
        self._ensure_class_dirs()
        dest_dir = Path(self.save_folder_path) / CLASS_SAVE_DIRS[slug]
        dest = dest_dir / self._crop_basename(idx)

        old = self.contour_saved_paths.get(idx)
        if old is not None and old != dest:
            try:
                if old.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest.resolve() != old.resolve():
                        dest.unlink()
                    shutil.move(str(old), str(dest))
                    self.contour_saved_paths[idx] = dest
                    try:
                        rel = dest.relative_to(self.save_folder_path)
                    except ValueError:
                        rel = dest
                    if not skip_review_refresh:
                        self.status_var.set(f"Flake #{idx + 1}: moved → {rel}")
                        self._review_refresh_list()
                    return
            except OSError as e:
                if not skip_review_refresh:
                    self.status_var.set(f"Flake #{idx + 1}: move failed ({e}); saving new file.")

        img_rgb = np.array(self.current_image)
        cnt = np.asarray(self.contours[idx], dtype=np.int32).reshape(-1, 1, 2)
        x, y, x2, y2 = self.find_contour_bounded_box(img_rgb, cnt)
        crop = img_rgb[y:y2, x:x2]
        if crop.size == 0:
            if not skip_review_refresh:
                self.status_var.set(f"Flake #{idx + 1}: crop empty; not saved.")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        self.contour_saved_paths[idx] = dest
        try:
            rel = dest.relative_to(self.save_folder_path)
        except ValueError:
            rel = dest
        if not skip_review_refresh:
            self.status_var.set(f"Flake #{idx + 1} → {self._class_title(slug)}  ({rel})")
            self._review_refresh_list()

    # ── Active class ──────────────────────────────────────────────────────────

    def set_active_class(self, slug: str):
        # Reset all buttons to normal style
        for s, btn in self.buttons.items():
            btn.configure(style=f"{_STYLE_ID[s]}.Class.TButton")
        # Armed style for selected
        self.buttons[slug].configure(style=f"{_STYLE_ID[slug]}Armed.Class.TButton")
        self.active_label = slug
        title = self._class_title(slug)
        self.active_label_var.set(f"Armed: {title}")
        self.status_var.set(f"Armed: {title} — click or drag to label")
        self.redraw_canvas()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def create_menu(self):
        menubar = tk.Menu(self.root, bg=T_CARD, fg=T_FG, activebackground=T_SURFACE)
        file_menu = tk.Menu(menubar, tearoff=0, bg=T_CARD, fg=T_FG,
                            activebackground=T_SURFACE, activeforeground=T_FG)
        file_menu.add_command(label="Open Raw Image…",                   command=self.open_image)
        file_menu.add_command(label="Open Folder… (Raw images, recursive)", command=self.open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Choose save folder…",               command=self.choose_save_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",                              command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    # ── File opening ──────────────────────────────────────────────────────────

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not file_path:
            return
        p = Path(file_path)
        if not self._is_under_raw_folder(p):
            self.status_var.set("Only files inside a Raw/ folder are supported.")
            return
        self.images.append(p)
        self._folder_for_image.append(None)
        self.image_index = len(self.images) - 1
        self._refresh_current_image_from_index()
        if len(self.images) > 1:
            self.next_image_button.state(["!disabled"])
            self.previous_image_button.state(["!disabled"])

    def _collect_images_recursive(self, folder: Path) -> list[Path]:
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        folder = Path(folder).resolve()
        out: list[Path] = []
        try:
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in valid_ext:
                    continue
                if not self._is_under_raw_folder(p):
                    continue
                try:
                    rel = p.relative_to(folder)
                except ValueError:
                    continue
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if "__pycache__" in rel.parts:
                    continue
                out.append(p)
        except OSError:
            return []
        out.sort(key=lambda x: str(x.relative_to(folder)).lower())
        return out

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        root = Path(folder_path).resolve()
        file_list = self._collect_images_recursive(root)
        if not file_list:
            self.status_var.set(
                f"No raw images under {root}. Files must live inside a folder named Raw/."
            )
            return
        start_index = len(self.images)
        for fp in file_list:
            self.images.append(fp)
            self._folder_for_image.append(root)
        self.image_index = start_index
        self._refresh_current_image_from_index()
        self.status_var.set(
            f"Loaded {len(file_list)} image(s) from {root.name}.  Save: {self.save_folder_path}"
        )
        self.next_image_button.state(["!disabled"])
        self.previous_image_button.state(["!disabled"])

    # ── Image loading ─────────────────────────────────────────────────────────

    def _refresh_current_image_from_index(self) -> None:
        p = Path(self.images[self.image_index])
        self.current_image = self._pil_for_labeling(p)
        root = self._folder_for_image[self.image_index]
        if root is not None:
            try:
                label = str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                label = p.name
        else:
            label = p.name
        self.image_name_var.set(label)
        n = len(self.images)
        self.image_counter_var.set(f"Image {self.image_index + 1} of {n}")
        self._load_contours_for_current_image()

    def _yellow_kwargs(self) -> dict:
        return dict(
            suppress_yellow=bool(self.suppress_yellow_var.get()),
            yellow_hue_lo=int(self.yellow_hue_lo_var.get()),
            yellow_hue_hi=int(self.yellow_hue_hi_var.get()),
            yellow_sat_min=int(self.yellow_sat_min_var.get()),
            yellow_val_min=int(self.yellow_val_min_var.get()),
            filter_yellow_interior=bool(self.filter_yellow_interior_var.get()),
            yellow_interior_pct=int(self.yellow_interior_pct_var.get()),
        )

    def _load_contours_for_current_image(self):
        if self.current_image is None:
            return
        img_bgr = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
        _bg, self.contours = flake_finder.find_flakes(img_bgr, display=False, **self._yellow_kwargs())

        # Start with blank labels, then restore from cache if available
        self.contour_labels = [None] * len(self.contours)
        self.contour_saved_paths.clear()
        self.hover_idx = None
        self._box_pending_rect = None
        self._box_pending_indices = []

        self._restore_labels_for_current()

        n = len(self.contours)
        self.contour_count_var.set(f"Found: {n}")
        self._update_labeled_count()
        self.redraw_canvas()

    def _reload_contours(self) -> None:
        """Re-run detection with updated yellow settings; preserve existing labels by index."""
        if self.current_image is None:
            return
        img_bgr = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
        _bg, new_contours = flake_finder.find_flakes(img_bgr, display=False, **self._yellow_kwargs())
        old_labels = self.contour_labels[:]
        old_paths  = dict(self.contour_saved_paths)
        self.contours = new_contours
        self.contour_labels = [
            old_labels[i] if i < len(old_labels) else None
            for i in range(len(new_contours))
        ]
        self.contour_saved_paths = {
            i: old_paths[i] for i in range(len(new_contours)) if i in old_paths
        }
        self.hover_idx = None
        self._box_pending_rect = None
        self._box_pending_indices = []
        self.contour_count_var.set(f"Found: {len(new_contours)}")
        self._update_labeled_count()
        self.redraw_canvas()

    def _update_labeled_count(self):
        n = len(self.contour_labels)
        k = sum(1 for x in self.contour_labels if x is not None)
        self.labeled_count_var.set(f"Labeled: {k} / {n}")

    # ── Navigation ────────────────────────────────────────────────────────────

    def next_image(self, event=None):
        if not self.images:
            return
        self._save_labels_for_current()
        self.image_index = (self.image_index + 1) % len(self.images)
        self._refresh_current_image_from_index()

    def previous_image(self, event=None):
        if not self.images:
            return
        self._save_labels_for_current()
        self.image_index = (self.image_index - 1) % len(self.images)
        self._refresh_current_image_from_index()

    # ── Display geometry ──────────────────────────────────────────────────────

    def _compute_display_geometry(self, cw: int, ch: int):
        if self.current_image is None or cw < 2 or ch < 2:
            return
        self._img_w, self._img_h = self.current_image.size
        iw, ih = self._img_w, self._img_h
        scale = min(cw / iw, ch / ih)
        self._disp_w = max(1, int(iw * scale))
        self._disp_h = max(1, int(ih * scale))
        self._off_x  = (cw - self._disp_w) // 2
        self._off_y  = (ch - self._disp_h) // 2

    def _canvas_to_image(self, mx: float, my: float) -> tuple[int, int] | None:
        if self._disp_w <= 0 or self._disp_h <= 0:
            return None
        lx = mx - self._off_x
        ly = my - self._off_y
        if lx < 0 or ly < 0 or lx >= self._disp_w or ly >= self._disp_h:
            return None
        ix = int(lx * self._img_w / self._disp_w)
        iy = int(ly * self._img_h / self._disp_h)
        ix = max(0, min(self._img_w - 1, ix))
        iy = max(0, min(self._img_h - 1, iy))
        return ix, iy

    def _pick_contour_at(self, ix: int, iy: int) -> int | None:
        candidates: list[tuple[float, int]] = []
        for i, c in enumerate(self.contours):
            if cv2.pointPolygonTest(c, (float(ix), float(iy)), False) >= 0:
                candidates.append((cv2.contourArea(c), i))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _contour_to_canvas_flat(self, contour) -> list[float]:
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        sx  = self._disp_w / self._img_w
        sy  = self._disp_h / self._img_h
        out: list[float] = []
        for x, y in pts:
            out.extend((float(x) * sx + self._off_x, float(y) * sy + self._off_y))
        return out

    def _bbox_to_canvas(self, x, y, x2, y2) -> tuple[float, float, float, float]:
        sx = self._disp_w / self._img_w
        sy = self._disp_h / self._img_h
        return (float(x)*sx+self._off_x, float(y)*sy+self._off_y,
                float(x2)*sx+self._off_x, float(y2)*sy+self._off_y)

    def _canvas_rect_to_image_rect(self, cx1, cy1, cx2, cy2) -> tuple[int,int,int,int] | None:
        if self.current_image is None or self._disp_w <= 0:
            return None
        x1, y1 = min(cx1, cx2), min(cy1, cy2)
        x2, y2 = max(cx1, cx2), max(cy1, cy2)
        x1 = max(x1, float(self._off_x));   y1 = max(y1, float(self._off_y))
        x2 = min(x2, float(self._off_x+self._disp_w))
        y2 = min(y2, float(self._off_y+self._disp_h))
        if x1 >= x2 or y1 >= y2:
            return None
        ix1 = int((x1-self._off_x)*self._img_w/self._disp_w)
        iy1 = int((y1-self._off_y)*self._img_h/self._disp_h)
        ix2 = int((x2-self._off_x)*self._img_w/self._disp_w)
        iy2 = int((y2-self._off_y)*self._img_h/self._disp_h)
        ix1 = max(0, min(self._img_w-1, ix1)); iy1 = max(0, min(self._img_h-1, iy1))
        ix2 = max(0, min(self._img_w-1, ix2)); iy2 = max(0, min(self._img_h-1, iy2))
        if ix1 > ix2: ix1, ix2 = ix2, ix1
        if iy1 > iy2: iy1, iy2 = iy2, iy1
        return (ix1, iy1, ix2, iy2)

    def _contours_overlapping_image_rect(self, rx1, ry1, rx2, ry2) -> list[int]:
        out: list[int] = []
        for i, cnt in enumerate(self.contours):
            x, y, w, h = cv2.boundingRect(np.asarray(cnt))
            if x+w <= rx1 or x > rx2 or y+h <= ry1 or y > ry2:
                continue
            out.append(i)
        return out

    # ── Box-select confirm / cancel ───────────────────────────────────────────

    def _confirm_box_selection(self, _event=None) -> None:
        if not self._box_pending_indices:
            return
        if self.active_label is None:
            self.status_var.set("Arm a class first, then press Enter.")
            return
        slug    = self.active_label
        indices = self._box_pending_indices
        n       = len(indices)
        for i, idx in enumerate(indices):
            self.contour_labels[idx] = slug
            self._save_crop_for_contour(idx, slug, skip_review_refresh=(i < n - 1))
        self._box_pending_rect    = None
        self._box_pending_indices = []
        self._update_labeled_count()
        self.redraw_canvas()
        self._review_refresh_list()
        self.status_var.set(f"Labeled {n} flake(s) as {self._class_title(slug)}")

    def _cancel_box_selection(self, _event=None) -> None:
        if self._box_pending_indices or self._box_pending_rect:
            self._box_pending_rect    = None
            self._box_pending_indices = []
            self.redraw_canvas()
            self.status_var.set("Box selection cancelled.")

    # ── Canvas grab helpers ───────────────────────────────────────────────────

    def _safe_grab_release(self) -> None:
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass

    def _grab_for_box_drag(self) -> None:
        try:
            self.canvas.grab_set_global()
        except tk.TclError:
            try:
                self.canvas.grab_set()
            except tk.TclError:
                pass

    def _event_to_canvas_xy(self, event) -> tuple[float, float]:
        w = getattr(event, "widget", None)
        if w is self.canvas:
            return float(event.x), float(event.y)
        try:
            return (float(event.x_root - self.canvas.winfo_rootx()),
                    float(event.y_root - self.canvas.winfo_rooty()))
        except tk.TclError:
            return float(getattr(event, "x", 0)), float(getattr(event, "y", 0))

    # ── Canvas events ─────────────────────────────────────────────────────────

    def _on_canvas_configure(self, _event):
        self.redraw_canvas()

    def _on_canvas_motion(self, event):
        if self._box_dragging:
            return
        if not self.contours or self.current_image is None:
            return
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            if self.hover_idx is not None:
                self.hover_idx = None
                self.hover_info_var.set("Hover: —")
                self.redraw_canvas()
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx != self.hover_idx:
            self.hover_idx = idx
            if idx is None:
                self.hover_info_var.set("Hover: —")
            else:
                lab = self.contour_labels[idx]
                t   = "unlabeled" if lab is None else self._class_title(lab)
                self.hover_info_var.set(f"Hover: #{idx + 1} ({t})")
            self.redraw_canvas()

    def _on_canvas_leave(self, _event):
        if self._box_dragging:
            return
        if self.hover_idx is not None:
            self.hover_idx = None
            self.hover_info_var.set("Hover: —")
            self.redraw_canvas()

    def _on_canvas_button_press(self, event):
        if int(self.box_select_var.get()) == 0:
            self._on_canvas_click(event)
            return
        if self.active_label is None:
            self.status_var.set("Arm a class (1–4) first, then drag a box.")
            return
        if not self.contours or self.current_image is None:
            return
        self._grab_for_box_drag()
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass
        mx, my = self._event_to_canvas_xy(event)
        self._box_dragging     = True
        self._box_drag_start   = (mx, my)
        self._box_drag_current = (mx, my)
        self.redraw_canvas()

    def _on_canvas_b1_motion(self, event):
        if not self._box_dragging:
            return
        mx, my = self._event_to_canvas_xy(event)
        self._box_drag_current = (mx, my)
        self.redraw_canvas()

    def _on_canvas_button_release(self, event):
        self._safe_grab_release()
        start      = self._box_drag_start
        was_dragging = self._box_dragging
        self._box_dragging     = False
        self._box_drag_start   = None
        self._box_drag_current = None
        self.redraw_canvas()

        if not was_dragging or start is None:
            return
        if int(self.box_select_var.get()) == 0:
            return
        end = self._event_to_canvas_xy(event)
        dx  = abs(end[0] - start[0])
        dy  = abs(end[1] - start[1])
        if dx < BOX_SELECT_MIN_DRAG and dy < BOX_SELECT_MIN_DRAG:
            self._on_canvas_click(event)
            return
        ir = self._canvas_rect_to_image_rect(start[0], start[1], end[0], end[1])
        if ir is None:
            self.status_var.set("Drag inside the image.")
            return
        rx1, ry1, rx2, ry2 = ir
        indices = self._contours_overlapping_image_rect(rx1, ry1, rx2, ry2)
        if not indices:
            self.status_var.set("No flakes in that box.")
            return
        self._box_pending_rect    = (start[0], start[1], end[0], end[1])
        self._box_pending_indices = indices
        self.redraw_canvas()
        title = self._class_title(self.active_label) if self.active_label else "?"
        self.status_var.set(
            f"{len(indices)} flake(s) selected as {title} — Enter to confirm, Esc to cancel"
        )

    def _on_canvas_click(self, event):
        if self.active_label is None:
            self.status_var.set("Choose a class (1–4) first, then click a flake.")
            return
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx is None:
            return
        self.contour_labels[idx] = self.active_label
        self._save_crop_for_contour(idx, self.active_label)
        self._update_labeled_count()
        self.redraw_canvas()

    def _on_canvas_right_click(self, event):
        pos = self._canvas_to_image(event.x, event.y)
        if pos is None:
            return
        idx = self._pick_contour_at(pos[0], pos[1])
        if idx is None:
            return
        self.contour_labels[idx] = None
        old = self.contour_saved_paths.pop(idx, None)
        if old is not None:
            try:
                if old.exists():
                    old.unlink()
            except OSError:
                pass
        self._update_labeled_count()
        self.redraw_canvas()
        self._review_refresh_list()
        self.status_var.set(f"Cleared label on flake #{idx + 1}")

    # ── Main canvas draw ──────────────────────────────────────────────────────

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.current_image is None:
            return

        self.right_placeholder.place_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True)

        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            self.root.after(50, self.redraw_canvas)
            return

        self._compute_display_geometry(cw, ch)
        disp = self.current_image.resize((self._disp_w, self._disp_h), Image.Resampling.LANCZOS)

        # Filled-contour overlay
        opacity = float(self.overlay_opacity_var.get())
        if opacity > 0.0 and self.contours:
            disp_np = np.array(disp)
            overlay = np.zeros_like(disp_np)
            sx = self._disp_w / self._img_w
            sy = self._disp_h / self._img_h
            for i, cnt in enumerate(self.contours):
                slug = self.contour_labels[i]
                ok   = self._outline_for_slug(slug)
                if slug is None:
                    if int(self.show_unlabeled_var.get()) == 0:
                        continue
                    fill_hex = SEGMENTATION_LINE_HEX
                else:
                    if ok is None or int(self.filter_vars[ok].get()) == 0:
                        continue
                    fill_hex = OUTLINE_HEX[ok]
                r = int(fill_hex[1:3], 16)
                g = int(fill_hex[3:5], 16)
                b = int(fill_hex[5:7], 16)
                pts = (np.asarray(cnt, dtype=np.float64).reshape(-1, 2) * [sx, sy]).astype(np.int32)
                cv2.fillPoly(overlay, [pts], (r, g, b))
            mask = overlay.sum(axis=2) > 0
            disp_np[mask] = (disp_np[mask]*(1.0-opacity) + overlay[mask]*opacity).astype(np.uint8)
            disp = Image.fromarray(disp_np)

        self._tk_image = ImageTk.PhotoImage(disp)
        self.canvas.create_image(
            self._off_x + self._disp_w // 2,
            self._off_y + self._disp_h // 2,
            image=self._tk_image,
        )

        img_rgb   = np.array(self.current_image)
        hovered   = self.hover_idx
        hover_only = int(self.hover_only_var.get()) != 0

        for i, cnt in enumerate(self.contours):
            slug = self.contour_labels[i]
            ok   = self._outline_for_slug(slug)

            if slug is None:
                if int(self.show_unlabeled_var.get()) == 0:
                    continue
            else:
                if ok is None or int(self.filter_vars[ok].get()) == 0:
                    continue

            if hover_only and (hovered is None or i != hovered):
                continue

            is_hov = (i == hovered)
            lw     = CONTOUR_WIDTH_HOVER if is_hov else CONTOUR_WIDTH
            flat   = self._contour_to_canvas_flat(cnt)
            if len(flat) >= 6:
                self.canvas.create_polygon(*flat, outline=SEGMENTATION_LINE_HEX,
                                           fill="", width=lw, smooth=False)

            if slug is not None and ok is not None and int(self.filter_vars[ok].get()) != 0:
                cnt_i = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
                x, y, x2, y2 = self.find_contour_bounded_box(img_rgb, cnt_i)
                bx1, by1, bx2, by2 = self._bbox_to_canvas(x, y, x2, y2)
                bw = BBOX_WIDTH_HOVER if is_hov else BBOX_WIDTH
                self.canvas.create_rectangle(bx1, by1, bx2, by2,
                                             outline=OUTLINE_HEX[ok], width=bw, fill="")

        preview_color = (
            OUTLINE_HEX[self._outline_for_slug(self.active_label)]
            if self.active_label is not None else T_ACCENT
        )

        if self._box_dragging and self._box_drag_start and self._box_drag_current:
            x1, y1 = self._box_drag_start
            x2, y2 = self._box_drag_current
            ir = self._canvas_rect_to_image_rect(x1, y1, x2, y2)
            if ir is not None:
                rx1, ry1, rx2, ry2 = ir
                for i, cnt in enumerate(self.contours):
                    cx, cy, ccw, cch = cv2.boundingRect(np.asarray(cnt))
                    if cx+ccw <= rx1 or cx > rx2 or cy+cch <= ry1 or cy > ry2:
                        continue
                    cnt_i = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
                    bx1b, by1b, bx2b, by2b = self._bbox_to_canvas(*self.find_contour_bounded_box(img_rgb, cnt_i))
                    self.canvas.create_rectangle(bx1b, by1b, bx2b, by2b,
                                                 outline=preview_color, width=BBOX_WIDTH_HOVER, fill="")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=T_ACCENT, width=2, dash=(5, 4))

        elif self._box_pending_rect is not None and self._box_pending_indices:
            x1, y1, x2, y2 = self._box_pending_rect
            for idx in self._box_pending_indices:
                cnt_i = np.asarray(self.contours[idx], dtype=np.int32).reshape(-1, 1, 2)
                bx1, by1, bx2, by2 = self._bbox_to_canvas(*self.find_contour_bounded_box(img_rgb, cnt_i))
                self.canvas.create_rectangle(bx1, by1, bx2, by2,
                                             outline=preview_color, width=BBOX_WIDTH_HOVER, fill="")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=T_ACCENT, width=2, dash=(5, 4))

        # Armed class badge — top-right corner of canvas
        if self.active_label is not None:
            color = OUTLINE_HEX[self.active_label]
            title = self._class_title(self.active_label)
            badge_text = f"  ● {title}  "
            bx = cw - 8
            by = 8
            self.canvas.create_text(bx, by, text=badge_text, anchor="ne",
                                    fill=color, font=("TkDefaultFont", 10, "bold"))

    # ── Contour utility ───────────────────────────────────────────────────────

    def find_contour_bounded_box(self, img: np.ndarray, contour):
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy     = x + w / 2, y + h / 2
        nw, nh     = w * CROP_SCALE, h * CROP_SCALE
        nx  = max(0, int(cx - nw / 2))
        ny  = max(0, int(cy - nh / 2))
        nx2 = min(img.shape[1], int(cx + nw / 2))
        ny2 = min(img.shape[0], int(cy + nh / 2))
        return nx, ny, nx2, ny2

    # ── Review strip ──────────────────────────────────────────────────────────

    def _review_on_folder_change(self) -> None:
        self.review_index = 0
        self._review_refresh_list()

    def _review_refresh_list(self) -> None:
        if not hasattr(self, "review_folder_var"):
            return
        dname  = self.review_folder_var.get()
        folder = Path(self.save_folder_path) / dname
        exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not folder.is_dir():
            self.review_paths = []
            self.review_index = 0
        else:
            self.review_paths = sorted(
                [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
                key=lambda p: p.name.lower(),
            )
        if self.review_index >= len(self.review_paths):
            self.review_index = max(0, len(self.review_paths) - 1)
        n = len(self.review_paths)
        if hasattr(self, "review_idx_var"):
            self.review_idx_var.set(f"{self.review_index + 1} / {n}" if n else "0 / 0")
        self._review_update_count_badge()
        win_open = (self._review_win is not None
                    and self._review_win.winfo_exists()
                    and self._review_win.state() != "withdrawn")
        if win_open:
            self._review_show_current()

    def _on_review_canvas_configure(self, _event=None) -> None:
        self._review_show_current()

    def _review_show_current(self) -> None:
        if not hasattr(self, "review_canvas") or not self.review_canvas.winfo_exists():
            return
        self.review_canvas.delete("all")
        if not self.review_paths:
            self.review_path_var.set("—")
            self._review_tk_img = None
            return
        p = self.review_paths[self.review_index]
        self.review_path_var.set(p.name)
        self.review_idx_var.set(f"{self.review_index + 1} / {len(self.review_paths)}")
        try:
            im = Image.open(p).convert("RGB")
        except OSError:
            self.review_path_var.set(f"(unreadable) {p.name}")
            self._review_tk_img = None
            return
        cw = max(60, self.review_canvas.winfo_width())
        ch = max(60, self.review_canvas.winfo_height())
        iw, ih = im.size
        if iw <= 0 or ih <= 0:
            return
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        disp = im.resize((nw, nh), Image.Resampling.LANCZOS)
        self._review_tk_img = ImageTk.PhotoImage(disp)
        self.review_canvas.create_image(cw // 2, ch // 2, image=self._review_tk_img)

    def _review_prev(self) -> None:
        if not self.review_paths:
            return
        self.review_index = (self.review_index - 1) % len(self.review_paths)
        self._review_show_current()

    def _review_next(self) -> None:
        if not self.review_paths:
            return
        self.review_index = (self.review_index + 1) % len(self.review_paths)
        self._review_show_current()

    def _review_move_to_slug(self, slug: str) -> None:
        if not self.review_paths:
            return
        p        = self.review_paths[self.review_index]
        dest_dir = Path(self.save_folder_path) / CLASS_SAVE_DIRS[slug]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if p.resolve() == dest.resolve():
            self.status_var.set("Already in that folder.")
            return
        try:
            if dest.exists():
                dest = dest_dir / f"{p.stem}_r{datetime.now().strftime('%H%M%S')}{p.suffix}"
            shutil.move(str(p), str(dest))
        except OSError as e:
            self.status_var.set(f"Move failed: {e}")
            return
        self.status_var.set(f"Moved to {CLASS_SAVE_DIRS[slug]}/{dest.name}")
        self._review_refresh_list()

    # ── Save folder ───────────────────────────────────────────────────────────

    def choose_save_folder(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self.save_folder_path = Path(path)
        self.save_folder_path.mkdir(parents=True, exist_ok=True)
        self._ensure_class_dirs()
        self.contour_saved_paths.clear()
        self.status_var.set(f"Save folder: {self.save_folder_path}")
        self._review_refresh_list()

    def on_resize(self, _event):
        if self.current_image is not None:
            self.redraw_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    DataLabelingApp2(root)
    root.mainloop()
