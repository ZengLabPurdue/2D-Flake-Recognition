import time

import cv2
import numpy as np
import tkinter as tk

from config import PIXEL_SIZE, CROP_RATIO, RESOLUTION_DIM
from Scanning.coordinate_generator import generate_rect_coords
from Scanning import wafer_detection

class Mapper:

    def __init__(
        self,
        root,
        app,
        stage,
        turret_controller,
        frame_processor,
        update_scan_status,
    ):
        self.root = root
        self.app = app
        self.stage = stage
        self.turret_controller = turret_controller
        self.frame_processor = frame_processor
        self.update_scan_status = update_scan_status

    def initialize_2x_mapping(self):

        self.app.set_live_mapping(False)

        self.turret_controller.change_objective(1)

        self.app.set_true_map(np.zeros((6000, 6000, 3), dtype=np.uint8))

        self.stage_center_x, self.stage_center_y, _ = self.stage.get_position()

        self.map_center_x = self.app.get_true_map().shape[1] // 2
        self.map_center_y = self.app.get_true_map().shape[0] // 2

        self.last_live_frame_rgb = None
        self.last_live_map_x = None
        self.last_live_map_y = None

        self.was_busy = False
        self.capture_after_move = False

        self.app.display_map()

    def set_live_map_2x(self):

        self.initialize_2x_mapping()

        self.app.set_live_mapping(True)

        self.app.open_panel("Stage Control Panel")

        self.app.set_view("Map", False)

    def place_frame_on_map(self, img, zoom, filter_img=None):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        step = max(1, int(round(zoom)))
        img_rgb = img_rgb[::step, ::step]

        h, w = img_rgb.shape[:2]

        cur_X, cur_Y, _ = self.stage.get_position()
        dx_um = cur_X - self.stage_center_x
        dy_um = cur_Y - self.stage_center_y

        map_pixels_per_um = 1 / PIXEL_SIZE[self.app.get_magnification()][self.app.get_resolution()] / zoom

        dx_px = -int(dx_um * map_pixels_per_um)
        dy_px = -int(dy_um * map_pixels_per_um)

        map_x = self.map_center_x + dx_px
        map_y = self.map_center_y + dy_px

        if self.last_live_frame_rgb is not None:
        
            try:
                last_frame = self.last_live_frame_rgb

                h_common = min(last_frame.shape[0], img_rgb.shape[0])
                w_common = min(last_frame.shape[1], img_rgb.shape[1])

                last_crop = last_frame[:h_common, :w_common]
                cur_crop = img_rgb[:h_common, :w_common]

                print("Running ORB/RANSAC shift correction...")
                shift_x, shift_y, score, num_matches, num_inliers = self.orb_ransac_shift_correction(
                    last_crop,
                    cur_crop
                )
                print("Finished ORB/RANSAC shift correction")

                print(
                    f"ORB/RANSAC Relative Shift: ({shift_x:.2f}, {shift_y:.2f}) | "
                    f"Score={score:.3f} | Matches={num_matches} | Inliers={num_inliers}"
                )

                '''
                self.save_shift_comparison(
                    last_crop,
                    cur_crop,
                    score=score,
                    shift_x=shift_x,
                    shift_y=shift_y,
                    filename=f"orb_compare_{int(time.time() * 1000)}.png"
                )
                '''

                map_x = self.last_live_map_x + int(round(shift_x))
                map_y = self.last_live_map_y + int(round(shift_y))

                # Add shift check here

            except Exception as e:
                print("Error during ORB/RANSAC shift correction:", e)

        else:
            print("Skipping correction: no previous live frame yet")

        true_map = self.app.get_true_map()
        filter_map = self.app.get_filter_map()
            
        x_start = int(map_x - w // 2)
        y_start = int(map_y - h // 2)

        x_end = x_start + w
        y_end = y_start + h

        x0 = max(0, x_start)
        y0 = max(0, y_start)

        x1 = min(true_map.shape[1], x_end)
        y1 = min(true_map.shape[0], y_end)

        crop_x0 = x0 - x_start
        crop_y0 = y0 - y_start

        crop_x1 = crop_x0 + (x1 - x0)
        crop_y1 = crop_y0 + (y1 - y0)

        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return

        crop = img_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
        if filter_img is not None:
            filter_crop = filter_img[crop_y0:crop_y1, crop_x0:crop_x1]

        existing = true_map[y0:y1, x0:x1]

        blended = cv2.addWeighted(existing, 0.5, crop, 0.5, 0)
        
        true_map[y0:y1, x0:x1] = crop
        if filter_img is not None:
            filter_map[y0:y1, x0:x1] = filter_crop

        self.app.set_true_map(true_map)

        self.last_live_frame_rgb = img_rgb.copy()
        self.last_live_map_x = map_x
        self.last_live_map_y = map_y

        self.app.display_map()

    def orb_ransac_shift_correction(self, existing, incoming, max_features=3000, min_matches=12, good_match_percent=0.35, ransac_reproj_threshold=5.0):

        existing_gray = cv2.cvtColor(existing, cv2.COLOR_RGB2GRAY)
        incoming_gray = cv2.cvtColor(incoming, cv2.COLOR_RGB2GRAY)

        existing_proc = self.preprocess_for_orb(existing_gray)
        incoming_proc = self.preprocess_for_orb(incoming_gray)

        orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=10
        )

        kp1, des1 = orb.detectAndCompute(existing_proc, None)
        kp2, des2 = orb.detectAndCompute(incoming_proc, None)

        if des1 is None or des2 is None:
            print("ORB failed: no descriptors found")
            return 0.0, 0.0, 0.0, 0, 0

        if len(kp1) < min_matches or len(kp2) < min_matches:
            print(f"ORB failed: not enough keypoints, kp1={len(kp1)}, kp2={len(kp2)}")
            return 0.0, 0.0, 0.0, 0, 0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)

        if len(matches) < min_matches:
            print(f"ORB failed: not enough matches, matches={len(matches)}")
            return 0.0, 0.0, 0.0, len(matches), 0

        matches = sorted(matches, key=lambda m: m.distance)

        keep_count = max(min_matches, int(len(matches) * good_match_percent))
        matches = matches[:keep_count]

        pts_existing = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts_incoming = np.float32([kp2[m.trainIdx].pt for m in matches])

        M, inliers = cv2.estimateAffinePartial2D(
            pts_incoming,
            pts_existing,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
            maxIters=2000,
            confidence=0.99
        )

        if M is None or inliers is None:
            print("ORB/RANSAC failed: affine transform could not be estimated")
            return 0.0, 0.0, 0.0, len(matches), 0

        num_inliers = int(inliers.sum())
        num_matches = len(matches)

        score = num_inliers / max(num_matches, 1)

        dx = float(M[0, 2])
        dy = float(M[1, 2])

        return dx, dy, score, num_matches, num_inliers

    def preprocess_for_orb(self, gray):

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        bg = cv2.GaussianBlur(gray, (0, 0), 35)
        flat = cv2.divide(gray, bg, scale=255)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        flat = clahe.apply(flat)

        flat = cv2.GaussianBlur(flat, (3, 3), 0)

        return flat

    def save_shift_comparison(self, existing_crop, img_rgb, score=None, shift_x=None, shift_y=None, save_dir=None, filename=None):

        existing_bgr = cv2.cvtColor(existing_crop, cv2.COLOR_RGB2BGR)
        live_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h = min(existing_bgr.shape[0], live_bgr.shape[0])
        w = min(existing_bgr.shape[1], live_bgr.shape[1])

        existing_bgr = existing_bgr[:h, :w]
        live_bgr = live_bgr[:h, :w]

        if shift_x is not None and shift_y is not None:
            warp = np.array([[1, 0, shift_x], [0, 1, shift_y]],dtype=np.float32)

            aligned_live_bgr = cv2.warpAffine(live_bgr, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            overlay = cv2.addWeighted(existing_bgr, 0.5, aligned_live_bgr, 0.5, 0)

        label_height = 90

        def add_label(img, title, extra_lines=None):
            labeled = cv2.copyMakeBorder( img, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            cv2.putText(labeled, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            if extra_lines:
                y = 58
                for line in extra_lines:
                    cv2.putText(labeled, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 24

            return labeled

        score_text = "Score: N/A" if score is None else f"Score: {score:.3f}"
        shift_text = "Shift: N/A" if shift_x is None or shift_y is None else f"Shift: dx={shift_x:.2f}, dy={shift_y:.2f}"

        existing_labeled = add_label(existing_bgr, "Existing Map Crop")

        live_labeled = add_label(live_bgr, "Live Frame")

        overlay_labeled = add_label(overlay, "Overlay: Existing + Live", [score_text, shift_text])

        comparison = np.hstack((existing_labeled, live_labeled, overlay_labeled))

        if filename is None:
            filename = f"ecc_compare_{int(time.time() * 1000)}.png"

        self.frame_processor.save_image(image=comparison, save_dir=save_dir, filename=filename, output=False)

    def auto_map_2x(self, window=(5, 5), zoom=3, full_zoom=True):

        self.scan_running = True

        camera_width, camera_height = self.frame_processor.get_camera().get_Size()

        #zoom = max(zoom, int(camera_height / (self.app.get_true_map().shape[0] / window[1])), int(camera_width / (self.app.get_true_map().shape[1] / window[0])))

        #if full_zoom:
        #    zoom = max(int(camera_height / (self.app.get_true_map().shape[0] / window[1])), int(camera_width / (self.app.get_true_map().shape[1] / window[0])))

        try:
            self.app.set_view("Map", True)

            self.stage.set_origin()

            self.initialize_2x_mapping()

            coords, total_frames = generate_rect_coords(window[1], window[0])

            print(f"Generated {total_frames} coordinates for mapping: {coords}")

            for i, (offset_x, offset_y) in enumerate(coords, start=1):

                resolution = self.app.get_resolution()

                target_x = offset_x * PIXEL_SIZE["2X"][resolution] * RESOLUTION_DIM[resolution]["x"] * CROP_RATIO["2X"]["x"] / 2
                target_y = -offset_y * PIXEL_SIZE["2X"][resolution] * RESOLUTION_DIM[resolution]["x"] * CROP_RATIO["2X"]["x"] / 2

                self.stage.move_to_xy(target_x, target_y)
                self.stage.wait_until_not_busy() 

                img = self.frame_processor.capture_frame(num_images=2)

                img = self.frame_processor.apply_vignette_filter(img)

                if img is None:
                    print("No image captured, skipping this tile.")
                    continue

                filter_img = wafer_detection.wafer_filter(img, display=False)

                #self.frame_processor.save_image(image=img, filename=f"mapper_output_{i}.png")

                self.place_frame_on_map(img, zoom=zoom, filter_img=filter_img)

                progress_percent = f"{i}/{total_frames}"

                self.update_scan_status(scan_type="Auto Map 2x", stage="Mapping", progress=progress_percent)

                self.root.update()

        except Exception as ex:
            print("Auto map error:", ex)

        finally:
            self.stage.move_to_xy(0, 0)
            self.app.set_live_mapping(False)
            self.app.set_view("Map", False)
            print("Auto mapping finished!")
