#!/usr/bin/env python3
"""Tkinter UI for reviewing 100x flake masks and exporting YOLO labels."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, messagebox, ttk

from export_100x_yolo_seg_labels import DEFAULT_OUT_DIR, IMAGE_EXTS, polygon_to_yolo_line


BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "images"
ANN_DIR = BASE / "contour_annotations"
MIN_POLY_POINTS = 3


def image_paths() -> list[Path]:
    if not IMG_DIR.exists():
        return []
    return sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ann_path(img_path: Path) -> Path:
    return ANN_DIR / f"{img_path.stem}.json"


def load_annotations(img_path: Path) -> list[list[tuple[int, int]]]:
    path = ann_path(img_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [[(int(x), int(y)) for x, y in poly] for poly in data.get("contours", [])]


def save_annotations(img_path: Path, polygons: list[list[tuple[int, int]]]) -> Path:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    path = ann_path(img_path)
    path.write_text(
        json.dumps(
            {
                "image": img_path.name,
                "contours": [[[int(x), int(y)] for x, y in poly] for poly in polygons],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def export_yolo(img_path: Path, polygons: list[list[tuple[int, int]]], out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    if not polygons:
        raise ValueError("No polygons to export")
    with Image.open(img_path) as img:
        width, height = img.size
    lines = [
        line
        for poly in polygons
        if (line := polygon_to_yolo_line(poly, width=width, height=height, class_id=0, decimals=6))
    ]
    if not lines:
        raise ValueError("No valid polygons to export")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_dir / img_path.name)
    label_path = out_dir / f"{img_path.stem}.txt"
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_path


def _bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    h2, w2 = h - h % factor, w - w % factor
    img = img[:h2, :w2]
    return np.stack([
        img[:, :, c].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3)).astype(np.uint8)
        for c in range(3)
    ], axis=2)


def _background_color_lab(img_lab: np.ndarray, corner_sz: int) -> np.ndarray:
    h, w = img_lab.shape[:2]
    csz = max(1, min(corner_sz, h // 4, w // 4))
    patches = [
        img_lab[:csz, :csz],
        img_lab[:csz, -csz:],
        img_lab[-csz:, :csz],
        img_lab[-csz:, -csz:],
    ]
    medians = [np.median(p.reshape(-1, 3), axis=0) for p in patches]
    best_pair, best_dist = (0, 1), float("inf")
    for i in range(4):
        for j in range(i + 1, 4):
            dist = float(np.linalg.norm(medians[i] - medians[j]))
            if dist < best_dist:
                best_pair, best_dist = (i, j), dist
    return np.mean([medians[best_pair[0]], medians[best_pair[1]]], axis=0)


def auto_detect_polygons(img_bgr: np.ndarray) -> list[list[tuple[int, int]]]:
    orig_h, orig_w = img_bgr.shape[:2]
    binned = _bin_image(img_bgr, 4)
    lab_bgr = cv2.cvtColor(binned, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_bgr[:, :, 0] = clahe.apply(lab_bgr[:, :, 0])
    enhanced = cv2.cvtColor(lab_bgr, cv2.COLOR_LAB2BGR)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = _background_color_lab(lab, 20)
    diff = np.linalg.norm(lab - bg_lab, axis=2).astype(np.float32)
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bin_h, bin_w = binned.shape[:2]
    sx, sy = orig_w / bin_w, orig_h / bin_h
    polygons: list[list[tuple[int, int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        poly = [
            (
                int(max(0, min(round(float(x) * sx), orig_w - 1))),
                int(max(0, min(round(float(y) * sy), orig_h - 1))),
            )
            for x, y in approx
        ]
        if len(poly) >= MIN_POLY_POINTS:
            polygons.append(poly)
    return polygons


class OutlineTool(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("100x CNN Mask Labeler")
        self.geometry("1300x850")
        self.paths = image_paths()
        self.idx = 0
        self.img_path: Path | None = None
        self.img_bgr: np.ndarray | None = None
        self.polygons: list[list[tuple[int, int]]] = []
        self.active: list[tuple[int, int]] = []
        self.scale = 1.0
        self.offset = (0, 0)
        self._photo = None
        self._build_ui()
        if self.paths:
            self.load_image(0)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(controls, text="Open Folder", command=self.open_folder).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Prev", command=lambda: self.load_image(self.idx - 1)).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Next", command=lambda: self.load_image(self.idx + 1)).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Auto Propose Contours", command=self.auto_propose).pack(fill=tk.X, pady=(16, 2))
        ttk.Button(controls, text="Save JSON", command=self.save_json).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Export YOLO Label", command=self.export_label).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Undo Point", command=self.undo_point).pack(fill=tk.X, pady=(16, 2))
        ttk.Button(controls, text="Finish Polygon", command=self.finish_polygon).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="Delete Last Polygon", command=self.delete_last).pack(fill=tk.X, pady=2)
        self.status = tk.StringVar(value="Put 100x images in 100xCNN/images")
        ttk.Label(controls, textvariable=self.status, wraplength=220).pack(fill=tk.X, pady=(16, 0))
        ttk.Label(
            controls,
            text="Left-click: add point\nDouble-click: finish\nRight-click: undo point",
            wraplength=220,
        ).pack(fill=tk.X, pady=(16, 0))

        self.canvas = tk.Canvas(self, bg="#111", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.finish_polygon())
        self.canvas.bind("<Button-3>", lambda _e: self.undo_point())
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(IMG_DIR))
        if not folder:
            return
        folder_path = Path(folder)
        self.paths = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if self.paths:
            self.load_image(0)

    def load_image(self, idx: int) -> None:
        if not self.paths:
            self.status.set("No images found")
            return
        if self.img_path is not None:
            self.save_json(silent=True)
        self.idx = max(0, min(idx, len(self.paths) - 1))
        self.img_path = self.paths[self.idx]
        self.img_bgr = cv2.imread(str(self.img_path), cv2.IMREAD_COLOR)
        if self.img_bgr is None:
            self.status.set(f"Could not load {self.img_path.name}")
            return
        self.polygons = load_annotations(self.img_path)
        self.active = []
        self.status.set(f"[{self.idx + 1}/{len(self.paths)}] {self.img_path.name} - {len(self.polygons)} contour(s)")
        self.redraw()

    def image_point(self, event) -> tuple[int, int]:
        ox, oy = self.offset
        x = int(round((event.x - ox) / max(self.scale, 1e-6)))
        y = int(round((event.y - oy) / max(self.scale, 1e-6)))
        if self.img_bgr is None:
            return x, y
        h, w = self.img_bgr.shape[:2]
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def canvas_point(self, x: int, y: int) -> tuple[float, float]:
        ox, oy = self.offset
        return x * self.scale + ox, y * self.scale + oy

    def add_point(self, event) -> None:
        if self.img_bgr is None:
            return
        self.active.append(self.image_point(event))
        self.status.set(f"Drawing polygon: {len(self.active)} point(s)")
        self.redraw()

    def finish_polygon(self) -> None:
        if len(self.active) < MIN_POLY_POINTS:
            self.status.set(f"Need at least {MIN_POLY_POINTS} points")
            return
        self.polygons.append(list(self.active))
        self.active = []
        self.status.set(f"Added polygon. Total: {len(self.polygons)}")
        self.redraw()

    def undo_point(self) -> None:
        if self.active:
            self.active.pop()
            self.redraw()

    def delete_last(self) -> None:
        if self.polygons:
            self.polygons.pop()
            self.status.set(f"Deleted last polygon. Total: {len(self.polygons)}")
            self.redraw()

    def auto_propose(self) -> None:
        if self.img_bgr is None:
            return
        proposed = auto_detect_polygons(self.img_bgr)
        if not proposed:
            self.status.set("Auto detector found no contours")
            return
        if self.polygons and not messagebox.askyesno("Replace contours?", "Replace current contours? Choose No to append."):
            self.polygons.extend(proposed)
        else:
            self.polygons = proposed
        self.active = []
        self.status.set(f"Auto proposed {len(proposed)} contour(s)")
        self.redraw()

    def save_json(self, silent: bool = False) -> None:
        if self.img_path is None:
            return
        path = save_annotations(self.img_path, self.polygons)
        if not silent:
            self.status.set(f"Saved {len(self.polygons)} contour(s) -> {path.name}")

    def export_label(self) -> None:
        if self.img_path is None:
            return
        try:
            save_annotations(self.img_path, self.polygons)
            label = export_yolo(self.img_path, self.polygons)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.status.set(f"Exported -> {label.relative_to(BASE)}")

    def redraw(self) -> None:
        if self.img_bgr is None:
            return
        cw, ch = self.canvas.winfo_width() or 900, self.canvas.winfo_height() or 700
        h, w = self.img_bgr.shape[:2]
        self.scale = min(cw / w, ch / h)
        dw, dh = int(w * self.scale), int(h * self.scale)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.offset = (ox, oy)

        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        canvas_img = Image.new("RGB", (cw, ch), (17, 17, 17))
        canvas_img.paste(pil, (ox, oy))
        draw = ImageDraw.Draw(canvas_img, "RGBA")

        for idx, poly in enumerate(self.polygons):
            pts = [self.canvas_point(x, y) for x, y in poly]
            if len(pts) >= 3:
                flat = [v for pt in pts for v in pt]
                draw.polygon(flat, fill=(50, 180, 255, 55))
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    draw.line([a, b], fill=(50, 180, 255, 230), width=2)
                draw.text(pts[0], f"#{idx + 1}", fill=(255, 255, 255, 240))

        if self.active:
            pts = [self.canvas_point(x, y) for x, y in self.active]
            for a, b in zip(pts, pts[1:]):
                draw.line([a, b], fill=(255, 80, 80, 255), width=2)
            for x, y in pts:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 80, 80, 255))

        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)


def main() -> int:
    OutlineTool().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
