"""
testyellow.py — side-by-side comparison of two yellow-filter approaches.

Panel layout (4 columns):
  1. Original
  2. Yellow mask highlight  (pixel-level HSV band)
  3. Pixel-mask filter      (zero out yellow pixels before Sobel)
  4. Contour-interior filter (drop whole contours where >50% interior pixels are yellow)

Run:
    python3.11 testyellow.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_EDGE_THRESHOLD  = 10
DEFAULT_AREA_THRESHOLD  = 500
DEFAULT_HUE_LO          = 15
DEFAULT_HUE_HI          = 38
DEFAULT_SAT_MIN         = 60
DEFAULT_VAL_MIN         = 80
DEFAULT_INTERIOR_THRESH = 50   # % of interior pixels that must be yellow to drop contour


# ── shared helpers ───────────────────────────────────────────────────────────

def _sobel_binary(bgr: np.ndarray, edge_thr: int) -> np.ndarray:
    R = bgr[:, :, 2].astype(np.float64)
    G = bgr[:, :, 1].astype(np.float64)
    stack = np.stack((R, G), axis=2)
    sm = cv2.GaussianBlur(stack, (5, 5), 0)
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(np.sum(gx ** 2 + gy ** 2, axis=2))
    return np.where(mag >= edge_thr, 255, 0).astype(np.uint8)


def _find_contours(binary: np.ndarray, area_thr: int) -> list:
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    raw, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in raw if cv2.contourArea(c) >= area_thr]


def _hsv_yellow_mask(bgr: np.ndarray, hue_lo: int, hue_hi: int,
                     sat_min: int, val_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((h >= hue_lo) & (h <= hue_hi) & (s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255


def _overlay_contours(rgb: np.ndarray, contours: list, color=(0, 255, 80)) -> np.ndarray:
    out = rgb.copy()
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ── Method A: pixel-mask (zero out yellow edge pixels before finding contours) ──

def run_pixel_mask(bgr, edge_thr, area_thr, hue_lo, hue_hi, sat_min, val_min, suppress):
    binary = _sobel_binary(bgr, edge_thr)
    if suppress:
        ymask = _hsv_yellow_mask(bgr, hue_lo, hue_hi, sat_min, val_min)
        binary[ymask > 0] = 0
    contours = _find_contours(binary, area_thr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _overlay_contours(rgb, contours), len(contours)


# ── Method B: contour-interior filter (from flake_extraction_pipeline.py) ───
# Drop any contour where >= threshold% of interior pixels fall in the yellow hue band.

def _filter_by_interior(contours: list, bgr: np.ndarray,
                        hue_lo: int, hue_hi: int, interior_pct: int) -> list:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    h, w = bgr.shape[:2]
    kept = []
    for c in contours:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        interior_H = H[mask > 127]
        n = len(interior_H)
        if n < 30:          # tiny contour — keep it
            kept.append(c)
            continue
        in_range = np.sum((interior_H >= hue_lo) & (interior_H <= hue_hi))
        if (in_range / n) < (interior_pct / 100.0):
            kept.append(c)
    return kept


def run_interior_filter(bgr, edge_thr, area_thr, hue_lo, hue_hi, interior_pct, suppress):
    binary = _sobel_binary(bgr, edge_thr)
    contours = _find_contours(binary, area_thr)
    if suppress:
        contours = _filter_by_interior(contours, bgr, hue_lo, hue_hi, interior_pct)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _overlay_contours(rgb, contours, color=(255, 100, 0)), len(contours)


# ── GUI ──────────────────────────────────────────────────────────────────────

class TestYellowApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Yellow filter comparison")
        self.root.minsize(1000, 560)
        self.root.geometry("1400x750")
        self.bgr: np.ndarray | None = None
        self._tk_imgs: list[ImageTk.PhotoImage] = []
        self._frames: list[np.ndarray] = []
        self._build_ui()

    def _build_ui(self):
        bg = "#ececf0"
        self.root.configure(bg=bg)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # ── left panel ──────────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=bg, width=270)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4), pady=10)
        left.pack_propagate(False)

        cb_kw = dict(bg=bg, activebackground=bg, selectcolor="white", anchor="w",
                     font=("TkDefaultFont", 9))

        tk.Button(left, text="Open image…", command=self._open_image,
                  font=("TkDefaultFont", 10, "bold")).pack(fill=tk.X, pady=(0, 8))

        self.status_var = tk.StringVar(value="No image loaded.")
        tk.Label(left, textvariable=self.status_var, bg=bg, wraplength=250,
                 justify=tk.LEFT, font=("TkDefaultFont", 9)).pack(anchor=tk.W, pady=(0, 8))

        self.suppress_var = tk.IntVar(value=1)
        tk.Checkbutton(left, text="Apply yellow filter", variable=self.suppress_var,
                       command=self._update, **cb_kw).pack(fill=tk.X, pady=(0, 6))

        def _sl(label, key, lo, hi, res, default):
            tk.Label(left, text=label, bg=bg, anchor="w",
                     font=("TkDefaultFont", 9)).pack(fill=tk.X)
            v = tk.IntVar(value=default)
            self._vars[key] = v
            tk.Scale(left, variable=v, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg=bg, troughcolor="#d1d5db",
                     highlightthickness=0, command=lambda _: self._update()
                     ).pack(fill=tk.X)

        self._vars: dict[str, tk.IntVar] = {}
        _sl("Edge threshold",          "edge_thr",     1,   60,  1, DEFAULT_EDGE_THRESHOLD)
        _sl("Area threshold",          "area_thr",    50, 5000, 50, DEFAULT_AREA_THRESHOLD)
        _sl("Hue lo (0–90)",           "hue_lo",       0,   90,  1, DEFAULT_HUE_LO)
        _sl("Hue hi (0–90)",           "hue_hi",       0,   90,  1, DEFAULT_HUE_HI)
        _sl("Sat min – pixel (0–255)", "sat_min",      0,  255,  1, DEFAULT_SAT_MIN)
        _sl("Val min – pixel (0–255)", "val_min",      0,  255,  1, DEFAULT_VAL_MIN)
        _sl("Interior % threshold",    "int_pct",      1,  100,  1, DEFAULT_INTERIOR_THRESH)

        tk.Button(left, text="Reset defaults", command=self._reset,
                  font=("TkDefaultFont", 9)).pack(fill=tk.X, pady=(10, 0))

        # ── right: 4-panel grid ──────────────────────────────────────────────
        right = tk.Frame(self.root, bg="#1a1a1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10, padx=(0, 10))

        panel_info = [
            ("Original",                     "#9ca3af"),
            ("Yellow mask (highlighted)",    "#fde68a"),
            ("A: Pixel-mask filter (green)", "#86efac"),
            ("B: Interior filter (orange)",  "#fdba74"),
        ]
        self.canvases: list[tk.Canvas] = []
        for i, (title, color) in enumerate(panel_info):
            col = tk.Frame(right, bg="#1a1a1e")
            col.grid(row=0, column=i, sticky="nsew", padx=3, pady=4)
            right.columnconfigure(i, weight=1)
            right.rowconfigure(0, weight=1)
            tk.Label(col, text=title, bg="#1a1a1e", fg=color,
                     font=("TkDefaultFont", 9, "bold"), wraplength=220).pack()
            c = tk.Canvas(col, bg="#121214", highlightthickness=0)
            c.pack(fill=tk.BOTH, expand=True)
            c.bind("<Configure>", lambda _e: self._redraw())
            self.canvases.append(c)

        self.count_vars = [tk.StringVar(value="") for _ in range(4)]
        # count labels below canvases
        for i, cv_ in enumerate(self.count_vars):
            tk.Label(right, textvariable=cv_, bg="#1a1a1e", fg="#9ca3af",
                     font=("TkDefaultFont", 8)).grid(row=1, column=i)

    # ── actions ──────────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.status_var.set(f"Could not read: {Path(path).name}")
            return
        self.bgr = bgr
        self.status_var.set(f"{Path(path).name}  ({bgr.shape[1]}×{bgr.shape[0]})")
        self._update()

    def _reset(self):
        for k, v in dict(edge_thr=DEFAULT_EDGE_THRESHOLD, area_thr=DEFAULT_AREA_THRESHOLD,
                         hue_lo=DEFAULT_HUE_LO, hue_hi=DEFAULT_HUE_HI,
                         sat_min=DEFAULT_SAT_MIN, val_min=DEFAULT_VAL_MIN,
                         int_pct=DEFAULT_INTERIOR_THRESH).items():
            self._vars[k].set(v)
        self._update()

    def _update(self):
        if self.bgr is None:
            return
        bgr = self.bgr
        et  = self._vars["edge_thr"].get()
        at  = self._vars["area_thr"].get()
        hlo = self._vars["hue_lo"].get()
        hhi = self._vars["hue_hi"].get()
        sm  = self._vars["sat_min"].get()
        vm  = self._vars["val_min"].get()
        ip  = self._vars["int_pct"].get()
        sup = bool(self.suppress_var.get())

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Panel 1: original
        # Panel 2: yellow mask highlight
        ymask = _hsv_yellow_mask(bgr, hlo, hhi, sm, vm)
        mask_vis = rgb.copy()
        ym = ymask > 0
        mask_vis[ym] = (mask_vis[ym] * 0.35 + np.array([255, 220, 0]) * 0.65).clip(0, 255).astype(np.uint8)

        # Panel 3: pixel-mask method
        pm_vis, n_pm = run_pixel_mask(bgr, et, at, hlo, hhi, sm, vm, sup)

        # Panel 4: contour-interior method
        ci_vis, n_ci = run_interior_filter(bgr, et, at, hlo, hhi, ip, sup)

        self._frames = [rgb, mask_vis, pm_vis, ci_vis]
        self.count_vars[0].set(f"{bgr.shape[1]}×{bgr.shape[0]}")
        self.count_vars[1].set(f"yellow px: {ym.sum():,}")
        self.count_vars[2].set(f"contours: {n_pm}")
        self.count_vars[3].set(f"contours: {n_ci}")
        self.status_var.set(
            f"{Path('?').name}  {bgr.shape[1]}×{bgr.shape[0]}  |  "
            f"pixel-mask: {n_pm}  interior: {n_ci}"
        )
        self._redraw()

    def _redraw(self):
        if not self._frames:
            return
        self._tk_imgs = []
        for canvas, frame in zip(self.canvases, self._frames):
            cw = max(2, canvas.winfo_width())
            ch = max(2, canvas.winfo_height())
            if cw < 10 or ch < 10:
                self.root.after(50, self._redraw)
                return
            ih, iw = frame.shape[:2]
            scale = min(cw / iw, ch / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            pil = Image.fromarray(frame).resize((nw, nh), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil)
            self._tk_imgs.append(tk_img)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=tk_img)


# ── entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    TestYellowApp(root)
    root.mainloop()
