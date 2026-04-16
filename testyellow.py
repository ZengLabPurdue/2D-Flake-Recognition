"""
testyellow.py — interactive yellow-filter + Sobel edge detection tester.

Open any image, tune the yellow HSV suppression sliders live, and see what
the edge detector picks up after masking yellow out.  Completely standalone;
nothing is saved and no labeling state is touched.

Run:
    python3.11 testyellow.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


# ── default Sobel / filter params ──────────────────────────────────────────
DEFAULT_EDGE_THRESHOLD = 10
DEFAULT_AREA_THRESHOLD = 500
DEFAULT_HUE_LO         = 15
DEFAULT_HUE_HI         = 38
DEFAULT_SAT_MIN        = 60
DEFAULT_VAL_MIN        = 80


# ── core detection ──────────────────────────────────────────────────────────

def _yellow_mask(bgr: np.ndarray, hue_lo: int, hue_hi: int,
                 sat_min: int, val_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((h >= hue_lo) & (h <= hue_hi) & (s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255


def _run_sobel(bgr: np.ndarray, edge_threshold: int, area_threshold: int,
               hue_lo: int, hue_hi: int, sat_min: int, val_min: int,
               suppress: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Returns (rgb_original, yellow_mask_rgb, edges_rgb, contours).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    R = bgr[:, :, 2].astype(np.float64)
    G = bgr[:, :, 1].astype(np.float64)
    stack = np.stack((R, G), axis=2)
    smoothed = cv2.GaussianBlur(stack, (5, 5), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(np.sum(gx ** 2 + gy ** 2, axis=2))
    binary = np.where(mag >= edge_threshold, 255, 0).astype(np.uint8)

    ymask = _yellow_mask(bgr, hue_lo, hue_hi, sat_min, val_min)
    if suppress:
        binary[ymask > 0] = 0

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours_raw, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours_raw if cv2.contourArea(c) >= area_threshold]

    # Yellow mask visualisation (tinted yellow)
    ymask_vis = rgb.copy()
    ym_bool = ymask > 0
    ymask_vis[ym_bool] = (ymask_vis[ym_bool] * 0.4 + np.array([255, 230, 0]) * 0.6).clip(0, 255).astype(np.uint8)

    # Edge overlay
    edge_vis = rgb.copy()
    cv2.drawContours(edge_vis, contours, -1, (0, 255, 80), 2)

    return rgb, ymask_vis, edge_vis, contours


# ── GUI ─────────────────────────────────────────────────────────────────────

class TestYellowApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Yellow filter tester")
        self.root.minsize(900, 560)
        self.root.geometry("1100x700")

        self.bgr: np.ndarray | None = None
        self._tk_imgs: list[ImageTk.PhotoImage] = []

        self._build_ui()

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        bg = "#ececf0"
        self.root.configure(bg=bg)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=bg)

        # ── left panel ──
        left = tk.Frame(self.root, bg=bg, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4), pady=10)
        left.pack_propagate(False)

        tk.Button(left, text="Open image…", command=self._open_image,
                  font=("TkDefaultFont", 10, "bold")).pack(fill=tk.X, pady=(0, 10))

        self.status_var = tk.StringVar(value="No image loaded.")
        tk.Label(left, textvariable=self.status_var, bg=bg, wraplength=240,
                 justify=tk.LEFT, font=("TkDefaultFont", 9)).pack(anchor=tk.W, pady=(0, 10))

        # suppress toggle
        self.suppress_var = tk.IntVar(value=1)
        tk.Checkbutton(left, text="Suppress yellow edges", variable=self.suppress_var,
                       bg=bg, activebackground=bg, selectcolor="white",
                       anchor="w", command=self._update,
                       font=("TkDefaultFont", 9)).pack(fill=tk.X, pady=(0, 6))

        sliders = [
            ("Edge threshold",  "edge_thr",  1,   60,  1, DEFAULT_EDGE_THRESHOLD),
            ("Area threshold",  "area_thr",  50, 5000, 50, DEFAULT_AREA_THRESHOLD),
            ("Hue lo (0–90)",   "hue_lo",    0,   90,  1, DEFAULT_HUE_LO),
            ("Hue hi (0–90)",   "hue_hi",    0,   90,  1, DEFAULT_HUE_HI),
            ("Sat min (0–255)", "sat_min",   0,  255,  1, DEFAULT_SAT_MIN),
            ("Val min (0–255)", "val_min",   0,  255,  1, DEFAULT_VAL_MIN),
        ]
        self._vars: dict[str, tk.IntVar] = {}
        for label, key, lo, hi, res, default in sliders:
            tk.Label(left, text=label, bg=bg, anchor="w",
                     font=("TkDefaultFont", 9)).pack(fill=tk.X)
            v = tk.IntVar(value=default)
            self._vars[key] = v
            tk.Scale(left, variable=v, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg=bg, troughcolor="#d1d5db",
                     highlightthickness=0, command=lambda _: self._update()
                     ).pack(fill=tk.X)

        tk.Button(left, text="Reset defaults", command=self._reset_defaults,
                  font=("TkDefaultFont", 9)).pack(fill=tk.X, pady=(10, 0))

        # ── right panel: three canvases ──
        right = tk.Frame(self.root, bg="#1a1a1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10, padx=(0, 10))

        titles = ["Original", "Yellow mask", "Edges after filter"]
        self.canvases: list[tk.Canvas] = []
        for i, title in enumerate(titles):
            col = tk.Frame(right, bg="#1a1a1e")
            col.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            right.columnconfigure(i, weight=1)
            right.rowconfigure(0, weight=1)
            tk.Label(col, text=title, bg="#1a1a1e", fg="#9ca3af",
                     font=("TkDefaultFont", 10, "bold")).pack()
            c = tk.Canvas(col, bg="#121214", highlightthickness=0)
            c.pack(fill=tk.BOTH, expand=True)
            c.bind("<Configure>", lambda _e: self._redraw())
            self.canvases.append(c)

    # ── actions ─────────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.status_var.set(f"Could not read: {Path(path).name}")
            return
        self.bgr = bgr
        self.status_var.set(f"{Path(path).name}  ({bgr.shape[1]}×{bgr.shape[0]})")
        self._update()

    def _reset_defaults(self):
        defaults = dict(edge_thr=DEFAULT_EDGE_THRESHOLD, area_thr=DEFAULT_AREA_THRESHOLD,
                        hue_lo=DEFAULT_HUE_LO, hue_hi=DEFAULT_HUE_HI,
                        sat_min=DEFAULT_SAT_MIN, val_min=DEFAULT_VAL_MIN)
        for k, v in defaults.items():
            self._vars[k].set(v)
        self._update()

    def _update(self):
        if self.bgr is None:
            return
        orig, ymask, edges, contours = _run_sobel(
            self.bgr,
            edge_threshold=self._vars["edge_thr"].get(),
            area_threshold=self._vars["area_thr"].get(),
            hue_lo=self._vars["hue_lo"].get(),
            hue_hi=self._vars["hue_hi"].get(),
            sat_min=self._vars["sat_min"].get(),
            val_min=self._vars["val_min"].get(),
            suppress=bool(self.suppress_var.get()),
        )
        self._frames = [orig, ymask, edges]
        self.status_var.set(
            f"{self.bgr.shape[1]}×{self.bgr.shape[0]}  |  "
            f"{len(contours)} contour(s) found"
        )
        self._redraw()

    def _redraw(self):
        if not hasattr(self, "_frames"):
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


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    TestYellowApp(root)
    root.mainloop()
