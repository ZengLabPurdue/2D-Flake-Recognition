# 100x CNN Mask Autodetector

Self-contained tools for building a 100x flake mask detector with YOLO segmentation.

Labels are **two-class segmentation**:

- class `0` = **good**
- class `1` = **bad**

Each contour polygon carries its own label in JSON, and exports into YOLO `.txt`
lines using those class ids.

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

   Pick **Good** vs **Bad** for the contour you are about to create.

   Check `SAM click mode`, then left-click a flake to have SAM propose its mask.
   Right-click adds an exclude point if SAM grabs too much. You can also leave
   SAM unchecked and draw polygons manually.

   To relabel an existing contour in manual mode: click near its edge to select it,
   choose Good/Bad, then press **Apply Label To Selected**.

   When the masks look right, use `Export Training Label`.

2. Export all saved JSON annotations:

   ```bash
   python3 export_100x_yolo_seg_labels.py
   ```

   Legacy JSON files without per-contour labels default to `--default-label good`.

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

`Export Training Label` writes the YOLO segmentation label format. YOLO here is
the CNN segmentation model being trained; the export step is just how masks get
saved for training.
