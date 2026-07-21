import json
import secrets
from pathlib import Path

import cv2
from Imaging import image_metadata
from Scanning.flake_identifier import Flake_Identifier


FLAKE_CLASSIFICATIONS = {
    0: "Bad Flake",
    1: "Good Flake",
    2: "Not a Flake",
    3: "Unclear Flake",
}


class Flake_Detector:

    def __init__(self):
        
        self.flake_identifier = Flake_Identifier()

    def flake_detection(
        self,
        image_queue,
        frame_processor,
        stop_requested=None,
        detection_model="Flake Detection",
        profile_path=None,
        scan_path=None,
    ):
        if detection_model not in ("Flake Detection", "Region Detection"):
            raise ValueError(f"Unknown detection model: {detection_model}")
        if detection_model == "Region Detection" and profile_path is None:
            raise ValueError("A scan profile is required for region detection.")

        scan_path = Path(scan_path) if scan_path is not None else None
        detection_records = []
        color_seed = secrets.randbits(32) if detection_model == "Region Detection" else None

        try:
            while True:
                image_data = image_queue.get()
                try:
                    if image_data is None:
                        break
                    if isinstance(image_data, dict):
                        img_path = Path(image_data["path"])
                    else:
                        img_path = Path(image_data)
                        image_data = {"path": img_path}

                    print(img_path)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    vignette_applied = image_metadata.is_vignette_corrected(img_path)
                    source_metadata = image_metadata.read_png_metadata(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if detection_model == "Region Detection":
                        scanned_img, detections, save, segmented_img = (
                            self.flake_identifier.identify_flakes_region_model(
                                img,
                                profile_path,
                                color_seed=color_seed,
                                return_segmented_map=True,
                            )
                        )
                    else:
                        scanned_img, detections, save = (
                            self.flake_identifier.identify_flakes_flake_model(img)
                        )
                        segmented_img = None

                    app = getattr(frame_processor, "app", None)
                    if (
                        segmented_img is not None
                        and app is not None
                        and hasattr(app, "place_region_map_frame")
                    ):
                        app.place_region_map_frame(
                            image_data.get("region_map_id"),
                            segmented_img,
                            image_data.get("map_x"),
                            image_data.get("map_y"),
                            image_data.get("map_zoom"),
                        )

                    for detection in detections:
                        record = self._create_detection_record(
                            detection,
                            detection_model,
                            image_data,
                            img.shape,
                            scan_path,
                        )
                        if record is not None:
                            detection_records.append(record)

                    out_path = img_path.parent.parent / "Processed" / img_path.name
                    frame_processor.save_image(
                        cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR),
                        save_dir=out_path.parent,
                        filename=out_path.name,
                        vignette_applied=vignette_applied,
                        metadata=source_metadata,
                    )

                    if save:
                        chip_folder = img_path.parent.parent
                        scan_root = scan_path or chip_folder.parent.parent.parent
                        flakes_dir = scan_root / "Flakes Found" / chip_folder.name
                        flakes_dir.mkdir(parents=True, exist_ok=True)
                        frame_processor.save_image(
                            cv2.cvtColor(scanned_img, cv2.COLOR_RGB2BGR),
                            save_dir=flakes_dir,
                            filename=img_path.name,
                            vignette_applied=vignette_applied,
                            metadata=source_metadata,
                        )
                finally:
                    image_queue.task_done()
        finally:
            if scan_path is not None:
                self._save_detection_records(
                    scan_path,
                    detection_model,
                    profile_path,
                    detection_records,
                    color_seed,
                )

    @staticmethod
    def _bounded_box(bounding_box, image_shape):
        if isinstance(bounding_box, dict):
            values = (
                bounding_box.get("x"),
                bounding_box.get("y"),
                bounding_box.get("width"),
                bounding_box.get("height"),
            )
        else:
            values = bounding_box

        if values is None or len(values) != 4:
            return None
        try:
            x, y, width, height = (float(value) for value in values)
        except (TypeError, ValueError):
            return None

        image_height, image_width = image_shape[:2]
        x1 = max(0.0, min(float(image_width), x))
        y1 = max(0.0, min(float(image_height), y))
        x2 = max(x1, min(float(image_width), x + max(0.0, width)))
        y2 = max(y1, min(float(image_height), y + max(0.0, height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def _create_detection_record(
        self,
        detection,
        detection_model,
        image_data,
        image_shape,
        scan_path,
    ):
        if detection_model == "Region Detection":
            classification = detection.get("matched_class")
            bounding_box = detection.get("bounding_box")
            class_id = None
        else:
            class_id, bounding_box = detection
            class_id = int(class_id)
            classification = FLAKE_CLASSIFICATIONS.get(
                class_id,
                f"Class {class_id}",
            )

        bounded_box = self._bounded_box(bounding_box, image_shape)
        if classification is None or bounded_box is None:
            return None

        x, y, width, height = bounded_box
        center_x = x + width / 2
        center_y = y + height / 2
        img_path = Path(image_data["path"])
        try:
            source_image = img_path.relative_to(scan_path).as_posix()
        except (TypeError, ValueError):
            source_image = str(img_path)

        record = {
            "classification": classification,
            "source_image": source_image,
            "magnification": image_data.get("magnification"),
            "bounding_box_px": {
                "x": round(x, 3),
                "y": round(y, 3),
                "width": round(width, 3),
                "height": round(height, 3),
            },
            "location": {
                "bounding_box_center_px": {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                },
            },
        }
        if class_id is not None:
            record["class_id"] = class_id

        stage_x = image_data.get("stage_x")
        stage_y = image_data.get("stage_y")
        pixel_size_um = image_data.get("pixel_size_um")
        if stage_x is not None and stage_y is not None and pixel_size_um is not None:
            image_height, image_width = image_shape[:2]
            target_x = float(stage_x) - (center_x - image_width / 2) * float(pixel_size_um)
            target_y = float(stage_y) - (center_y - image_height / 2) * float(pixel_size_um)
            record["location"]["stage_position_um"] = {
                "x": round(target_x, 6),
                "y": round(target_y, 6),
            }
            record["capture_stage_position_um"] = {
                "x": round(float(stage_x), 6),
                "y": round(float(stage_y), 6),
            }

        return record

    @staticmethod
    def _save_detection_records(
        scan_path,
        detection_model,
        profile_path,
        detection_records,
        color_seed=None,
    ):
        scan_path.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "flake-search.scan-detections",
            "version": 1,
            "detection_model": detection_model,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "region_color_seed": color_seed,
            "detections": detection_records,
        }
        (scan_path / "flakes_found.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
