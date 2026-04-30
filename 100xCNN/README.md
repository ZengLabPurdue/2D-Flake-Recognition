# 100x CNN Mask Autodetector

Self-contained tools for building a 100x flake mask detector with YOLO segmentation.

## Folder Layout

- `images/` - put 100x source images here.
- `contour_annotations/` - JSON polygon annotations from the outline UI.
- `training_images_100x/` - exported YOLO-seg image/label pairs.
- `labeled_seg_split_100x/` - generated train/val split.
- `segmentation_outputs_100x/` - inference overlays, masks, and CSV results.

## Workflow

1. Review or create masks:

   ```bash
   python3 outline_tool_100x.py
   ```

   Use `Auto Propose Contours`, correct the outlines, then `Export YOLO Label`.

2. Export all saved JSON annotations:

   ```bash
   python3 export_100x_yolo_seg_labels.py
   ```

3. Prepare a train/val split without training:

   ```bash
   python3 train_100x_segmentor.py --prepare-only
   ```

4. Train the 100x mask model:

   ```bash
   python3 train_100x_segmentor.py --epochs 150
   ```

5. Run detection:

   ```bash
   python3 run_segmentation_100x.py
   ```

The trained weights are copied to `flake_seg_100x_best.pt` in this folder.
