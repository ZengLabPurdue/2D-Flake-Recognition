import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path

import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from tkinter import filedialog

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model

from ultralytics import YOLO
os.environ["YOLO_VERBOSE"] = "False"

home_dir = os.path.dirname(os.path.abspath(__file__))
#brody_work_path = Path(home_dir) / "Brody's Work"
flake_reg_path = Path(home_dir) / "Flake Recognition"
color_model_path = Path(home_dir) / "color_classifier_tf.keras"
flake_model_path = Path(home_dir) / "flake_classifier_tf.keras"
#sys.path.insert(0, str(brody_work_path))
sys.path.insert(0, str(flake_reg_path))

#import single_frame_pipeline
import contour_finder
import flake_classifier

class Flake_Identifier():
    def __init__(self):
        try:
            #self.model = load_model(color_model_path)
            self.model = load_model(flake_model_path)
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")

        print("Flake identifier initialized!")

    # image should be in RGB
    def identify_flakes_color_model(self, image, output=False):
        start_time = time.time()
        masked_image, contours = contour_finder.find_flakes(image, display=False)

        valid_contours = []
        for c in contours:
            if isinstance(c, np.ndarray):
                try:
                    c_fixed = np.array(c, dtype=np.int32).reshape(-1,1,2)
                    valid_contours.append(c_fixed)
                except:
                    print("Skipping invalid contour")
            else:
                print("Skipping non-ndarray contour")

        scanned_image = image.copy()
        cv2.drawContours(scanned_image, valid_contours, -1, color=(255, 255, 255), thickness=2)
        save = False

        bounding_rectangles = []
        flakes = []

        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2, y + h / 2
            scale = 1.2
            new_w, new_h = w * scale, h * scale
            new_x = int(cx - new_w / 2)
            new_y = int(cy - new_h / 2)
            new_x2 = int(cx + new_w / 2)
            new_y2 = int(cy + new_h / 2)

            h_img, w_img = masked_image.shape[:2]

            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x2 = min(w_img, new_x2)
            new_y2 = min(h_img, new_y2)

            if new_x2 <= new_x or new_y2 <= new_y:
                continue

            bounding_rectangles.append((new_x, new_y, int(new_w), int(new_h)))

            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [c], -1, color=255, thickness=-1)
            contour_mask_crop = contour_mask[new_y:new_y2, new_x:new_x2]
            image_crop = image[new_y:new_y2, new_x:new_x2]
            comp_r = int(cv2.mean(image_crop[:, :, 0], mask=contour_mask_crop)[0])
            comp_g = int(cv2.mean(image_crop[:, :, 1], mask=contour_mask_crop)[0])
            comp_b = int(cv2.mean(image_crop[:, :, 2], mask=contour_mask_crop)[0])

            background_crop = masked_image[new_y:new_y2, new_x:new_x2]
            non_black_mask = cv2.cvtColor(background_crop, cv2.COLOR_RGB2GRAY) > 0
            back_r = int(background_crop[:, :, 0][non_black_mask].mean())
            back_g = int(background_crop[:, :, 1][non_black_mask].mean())
            back_b = int(background_crop[:, :, 2][non_black_mask].mean())

            flakes.append((c, (new_x, new_y, int(new_w), int(new_h)), (comp_r, comp_g, comp_b), (back_r, back_g, back_b)))

            if self.model:
                input_array = np.array([[comp_r, comp_g, comp_b, back_r, back_g, back_b]], dtype=np.float32) / 255.0
                try:
                    result = self.model.predict(input_array, verbose=0)
                    predicted_class = np.argmax(result, axis=1)[0]
                except:
                    predicted_class = 0

            class_to_color = {
                0: (200, 200, 200),  # Background - light gray
                1: (220, 180, 120),  # Glue - light brown/orange
                2: (0, 255, 0),      # Thin flake - green
                3: (0, 255, 200),    # Medium flake - teal
                4: (255, 255, 0),    # Thick flake - yellow
            }

            if predicted_class == 2:
                save = True

            rect_color = class_to_color.get(predicted_class, (255, 255, 255))
            cv2.rectangle(scanned_image, (new_x, new_y), (new_x2, new_y2), color=rect_color, thickness=2)
            cv2.rectangle(masked_image, (new_x, new_y), (new_x2, new_y2), color=rect_color, thickness=2)

        '''
        plt.figure(figsize=(10, 8))
        plt.imshow(masked_image)
        plt.title("Masked Image with Flake Colors")
        plt.axis("off")
        plt.show()
        '''

        if output:
            print(f"Time taken: {time.time() - start_time:.2f}s")        
            plt.figure(figsize=(10, 8))
            plt.imshow(scanned_image)
            plt.title("Image with Contours and Flake Colors")
            plt.axis("off")
            plt.show()
        
        return scanned_image, flakes, save
    
    def identify_flakes_flake_model(self, image, output=False, save_dir=None):
        start_time = time.time()

        save = False

        contours = self.find_flakes(image, output=False)
        
        red_green_values = []

        valid_contours = []
        for c in contours:
            try:
                c_fixed = np.array(c, dtype=np.int32).reshape(-1, 1, 2)
                valid_contours.append(c_fixed)
            except:
                continue
            
        scanned_image = image.copy()
        
        flakes = []
        
        class_to_color = {
            0: (255, 255, 0),    # Bad flake - yellow
            1: (0, 255, 0),      # Good flake - green
            2: (200, 200, 200),  # Not a flake - light gray
            3: (0, 255, 200),    # Unclear flake - teal
        }
        
        i = 0
        for c in valid_contours:
        
            x, y, w, h = cv2.boundingRect(c)
        
            cx, cy = x + w / 2, y + h / 2
        
            scale = 1.2
            new_w, new_h = w * scale, h * scale
        
            new_x = int(cx - new_w / 2)
            new_y = int(cy - new_h / 2)
            new_x2 = int(cx + new_w / 2)
            new_y2 = int(cy + new_h / 2)
        
            h_img, w_img = image.shape[:2]
        
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x2 = min(w_img, new_x2)
            new_y2 = min(h_img, new_y2)
        
            if new_x2 <= new_x or new_y2 <= new_y:
                continue
            
            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)

            crop = image[new_y:new_y2, new_x:new_x2]

            h_crop, w_crop = crop.shape[:2]

            if crop.size == 0:
                continue
            
            c_local = c.copy()
            c_local[:, 0, 0] -= new_x
            c_local[:, 0, 1] -= new_y

            mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
            cv2.drawContours(mask, [c_local], -1, 255, -1)

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            masked_pixels = gray_crop[mask == 255]

            if len(masked_pixels) == 0:
                continue
            
            '''
            avg_red_green = (float(crop[:, :, 0][mask == 255].mean()) + float(crop[:, :, 1][mask == 255].mean())) / 2

            if (avg_red_green > 150):
                continue
            '''
            
            i += 1
            crop = cv2.resize(crop, (flake_classifier.IMG_SIZE, flake_classifier.IMG_SIZE))
            crop = crop.astype(np.float32) / 255.0
            crop_filename = f"crop_{i}.png"

            crop_path = os.path.join(save_dir, crop_filename)

            plt.imsave(crop_path, crop)

            input_img = np.expand_dims(crop, axis=0)
        
            pred = self.model.predict(input_img, verbose=0)
            class_id = int(np.argmax(pred))
        
            color = class_to_color.get(class_id, (255, 255, 255))
        
            if class_id == 1:
                save = True

            cv2.rectangle(scanned_image,
                          (new_x, new_y),
                          (new_x2, new_y2),
                          color,
                          2)
        
            flakes.append((class_id, (new_x, new_y, new_w, new_h)))
        
        if output:
            print(f"Time: {time.time() - start_time:.2f}s")

            plt.figure(figsize=(10, 8))
            plt.imshow(scanned_image)
            plt.title("Flake Detection + Classification")
            plt.axis("off")
            plt.show()

        return scanned_image, flakes, save

    def find_flakes(self, image, output=False):

        _, contours = contour_finder.find_flakes(image, display=output)

        return contours
    
    def label_flakes(self, image, output=False, save_dir=None):

        types = []
        for i in range(100):
            types.append(0)
        types[5] = 3
        types[11] = 1
        types[22] = 3
        types[31] = 3
        types[40] = 3
        types[71] = 3
        types[78] = 3
        types[24] = 2
        types[28] = 2
        types[62] = 2

        class_to_color = {
            0: (255, 255, 0),    # Bad flake - yellow
            1: (0, 255, 0),      # Good flake - green
            2: (200, 200, 200),  # Not a flake - light gray
            3: (0, 255, 200),    # Unclear flake - teal
        }

        start_time = time.time()

        save = False

        contours = self.find_flakes(image, output=False)

        valid_contours = []
        for c in contours:
            try:
                c_fixed = np.array(c, dtype=np.int32).reshape(-1, 1, 2)
                valid_contours.append(c_fixed)
            except:
                continue
            
        scanned_image = image.copy()
        
        flakes = []
        
        i = 0
        for c in valid_contours:
        
            x, y, w, h = cv2.boundingRect(c)
        
            cx, cy = x + w / 2, y + h / 2
        
            scale = 1.2
            new_w, new_h = w * scale, h * scale
        
            new_x = int(cx - new_w / 2)
            new_y = int(cy - new_h / 2)
            new_x2 = int(cx + new_w / 2)
            new_y2 = int(cy + new_h / 2)
        
            h_img, w_img = image.shape[:2]
        
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_x2 = min(w_img, new_x2)
            new_y2 = min(h_img, new_y2)
        
            if new_x2 <= new_x or new_y2 <= new_y:
                continue
            
            #cv2.drawContours(scanned_image, [c], -1, (255, 255, 255), thickness=2)
    
            color = class_to_color.get(types[i], (255, 255, 255))
            i += 1

            cv2.rectangle(scanned_image,
                          (new_x, new_y),
                          (new_x2, new_y2),
                          color,
                          2)
        
        if output:
            print(f"Time: {time.time() - start_time:.2f}s")

            plt.figure(figsize=(10, 8))
            plt.imshow(scanned_image)
            plt.title("Flake Detection + Classification")
            plt.axis("off")
            plt.show()

        return scanned_image, flakes, save

if __name__ == "__main__":
    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    flake_id = Flake_Identifier()
    #save_dir = filedialog.askdirectory(title="Select Directory")
    #flake_id.identify_flakes_flake_model(image_rgb, output=True, save_dir=save_dir)

    flake_id.label_flakes(image_rgb, output=True)