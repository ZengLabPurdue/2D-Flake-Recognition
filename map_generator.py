import cv2
import numpy as np
from tkinter import filedialog

path1 = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
img1_color = cv2.imread(path1, cv2.IMREAD_COLOR)

path2 = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
img2_color = cv2.imread(path2, cv2.IMREAD_COLOR)

img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

win = cv2.createHanningWindow(img1.shape[::-1], cv2.CV_32F)

(dx, dy), response = cv2.phaseCorrelate(img1, img2, win)
print("Shift:", dx, dy, "Confidence:", response)

dx = int(round(dx))
dy = int(round(dy))

h1, w1 = img1_color.shape[:2]
h2, w2 = img2_color.shape[:2]

x2_rel = -dx
y2_rel = -dy

min_x = min(0, x2_rel)
min_y = min(0, y2_rel)
max_x = max(w1, x2_rel + w2)
max_y = max(h1, y2_rel + h2)

canvas_w = max_x - min_x
canvas_h = max_y - min_y

canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

offset_x = -min_x
offset_y = -min_y

# canvas[offset_y:offset_y + h1, offset_x:offset_x + w1] = img1_color

x2 = offset_x + x2_rel
y2 = offset_y + y2_rel

canvas[y2:y2 + h2, x2:x2 + w2] = img2_color

alpha = 0.9

roi = canvas[offset_y:offset_y + h1, offset_x:offset_x + w1]

mask = np.any(roi != 0, axis=2)

blended = roi.copy()

blended[mask] = cv2.addWeighted(img1_color[mask], alpha, roi[mask], 1 - alpha, 0)

blended[~mask] = img1_color[~mask]

canvas[offset_y:offset_y + h1, offset_x:offset_x + w1] = blended

cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)

while True:
    _, _, win_w, win_h = cv2.getWindowImageRect("Stitched")

    if win_w > 0 and win_h > 0:
        h, w = canvas.shape[:2]
        scale = min(win_w / w, win_h / h)
        display = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        cv2.imshow("Stitched", display)

    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()