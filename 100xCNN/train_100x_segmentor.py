#!/usr/bin/env python3
"""
Train a two-class YOLOv8 instance-segmentation model for 100x flake masks.

Input dataset:
    training_images_100x/

Each image needs a same-stem .txt YOLO segmentation label file.

Classes:
    0 good
    1 bad
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path


BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "training_images_100x"
SPLIT_DIR = BASE / "labeled_seg_split_100x"
PROJECT = BASE / "runs" / "segment"
RUN_NAME = "flake_seg_100x"
CLASSES = ["good", "bad"]
SAFE_OUT = BASE / "flake_seg_100x_best.pt"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_pairs(img_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for root, dirs, files in os.walk(img_dir):
        dirs.sort()
        for name in sorted(files):
            img = Path(root) / name
            if img.suffix.lower() not in IMG_EXTS:
                continue
            label = img.with_suffix(".txt")
            if label.exists() and label.stat().st_size > 0:
                pairs.append((img, label))
    return pairs


def build_split(
    pairs: list[tuple[Path, Path]],
    *,
    split_dir: Path,
    val_frac: float,
    seed: int,
) -> Path:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for subset in ("train", "val"):
        (split_dir / "images" / subset).mkdir(parents=True)
        (split_dir / "labels" / subset).mkdir(parents=True)

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        raise ValueError("No annotated image/label pairs")
    if n == 1:
        n_val = 0
    else:
        n_val = max(1, int(round(n * val_frac)))
        n_val = min(n_val, n - 1)

    val_indices = set(range(n_val))
    counts = {"train": 0, "val": 0}
    for idx, (img, label) in enumerate(shuffled):
        subset = "val" if idx in val_indices else "train"
        shutil.copy2(img, split_dir / "images" / subset / img.name)
        shutil.copy2(label, split_dir / "labels" / subset / label.name)
        counts[subset] += 1

    data_yaml = split_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {split_dir}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    print(f"Split -> train: {counts['train']}  val: {counts['val']}")
    return data_yaml


def train(args: argparse.Namespace) -> None:
    pairs = collect_pairs(args.img_dir)
    if not pairs:
        print(f"No annotated images found under {args.img_dir}")
        print("Run export_100x_yolo_seg_labels.py or export from outline_tool_100x.py first.")
        return

    print(f"Found {len(pairs)} annotated image/label pairs")
    data_yaml = build_split(
        pairs,
        split_dir=args.split_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    if args.prepare_only:
        print(f"Prepared split only: {data_yaml}")
        return

    from tqdm import tqdm
    from ultralytics import YOLO

    last_ckpt = args.project / args.run_name / "weights" / "last.pt"
    model = YOLO(str(last_ckpt) if args.resume and last_ckpt.exists() else args.model)

    state = {"pbar": None, "t": 0.0}

    def on_start(trainer):
        state["pbar"] = tqdm(total=trainer.epochs, desc="Training", unit="epoch")

    def on_epoch_start(_trainer):
        state["t"] = time.perf_counter()

    def on_epoch_end(trainer):
        pbar = state["pbar"]
        if not pbar:
            return
        metrics = trainer.metrics or {}
        map50 = metrics.get("metrics/mAP50(M)", metrics.get("metrics/mAP50", 0.0))
        pbar.set_postfix({"mAP50": f"{map50:.3f}", "s/ep": f"{time.perf_counter() - state['t']:.0f}"})
        pbar.update(1)

    def on_end(_trainer):
        if state["pbar"]:
            state["pbar"].close()

    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.add_callback("on_train_end", on_end)

    best = (args.project / args.run_name / "weights" / "best.pt").resolve()
    try:
        model.train(
            data=str(data_yaml.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.project.resolve()),
            name=args.run_name,
            exist_ok=True,
            resume=args.resume and last_ckpt.exists(),
            hsv_h=0.005,
            hsv_s=0.2,
            hsv_v=0.1,
            degrees=180,
            translate=0.1,
            scale=0.3,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            patience=30,
            save_period=5,
            verbose=False,
        )
    finally:
        if best.exists():
            shutil.copy2(best, args.safe_out)
            print(f"Best weights -> {args.safe_out}")
        else:
            print(f"WARNING: No checkpoint found at {best}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a 100x flake YOLO segmentation model.")
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--safe-out", type=Path, default=SAFE_OUT)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
