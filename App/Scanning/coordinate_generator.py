import math
import cv2
from config import PIXEL_SIZE, CROP_RATIO, RESOLUTION_DIM

def generate_rect_coords(x, y):
    rect_coords = []
    total_frames = x * y

    for i in range(x):
        if i % 2 == 0:
            y_range = range(y)
        else:
            y_range = range(y - 1, -1, -1)

        for j in y_range:
            rect_coords.append((i - x // 2, j - y // 2))

    return rect_coords, total_frames

def generate_spiral_coords(length):
    spiral_coords = []
    total_frames = length ** 2

    dx, dy = 0, 0
    step = 1
    direction = 0

    while len(spiral_coords) < total_frames:
        for _ in range(2):
            for _ in range(step):
                if len(spiral_coords) >= total_frames:
                    break

                spiral_coords.append((dx, dy))

                if direction == 0:
                    dx += 1
                elif direction == 1:
                    dy += 1
                elif direction == 2:
                    dx -= 1
                else:
                    dy -= 1

            direction = (direction + 1) % 4

        step += 1

    return spiral_coords, total_frames

def generate_10x_scan_coordinates(
    app,
    wafers,
    scan_center_x,
    scan_center_y,
    scale,
    true_map,
    camera_size,
):
    camera_width, camera_height = camera_size

    resolution = app.get_resolution()
    window_w = int(camera_width * CROP_RATIO["2X"]["x"] / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["2X"][resolution]) / CROP_RATIO["10X"]["x"])
    window_h = int(camera_height * CROP_RATIO["2X"]["y"] / scale / (PIXEL_SIZE["2X"][resolution] / PIXEL_SIZE["2X"][resolution]) / CROP_RATIO["10X"]["y"])

    scan_coordinates_10x = []

    for wafer in wafers:
        x, y, w, h = wafer

        num_windows_x = math.ceil(w / window_w)
        num_windows_y = math.ceil(h / window_h)

        grid_w = num_windows_x * window_w
        grid_h = num_windows_y * window_h

        wafer_center_x = x + w // 2
        wafer_center_y = y + h // 2

        start_x = max(0, wafer_center_x - grid_w // 2)
        start_y = max(0, wafer_center_y - grid_h // 2)

        start_pos_x = -(wafer_center_x - true_map.shape[1] / 2) * (PIXEL_SIZE["2X"][resolution] * RESOLUTION_DIM[resolution]["x"] * CROP_RATIO["2X"]["x"]) / (camera_width / scale * CROP_RATIO["2X"]["x"]) + scan_center_x
        start_pos_y = -(wafer_center_y - true_map.shape[0] / 2) * (PIXEL_SIZE["2X"][resolution] * RESOLUTION_DIM[resolution]["y"] * CROP_RATIO["2X"]["y"]) / (camera_height / scale * CROP_RATIO["2X"]["y"]) + scan_center_y

        scan_coordinates_10x.append([round(start_pos_x), round(start_pos_y), num_windows_x, num_windows_y,])

        for i in range(num_windows_x):
            for j in range(num_windows_y):
                wx = start_x + i * window_w
                wy = start_y + j * window_h

                cv2.rectangle(true_map, (wx, wy), (wx + window_w, wy + window_h), (0, 255, 0), 5, cv2.LINE_AA)

        cv2.circle(true_map, (wafer_center_x, wafer_center_y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(true_map, (int(true_map.shape[1] / 2), int(true_map.shape[0] / 2)), 8, (255, 0, 0), -1, cv2.LINE_AA)

        return scan_coordinates_10x