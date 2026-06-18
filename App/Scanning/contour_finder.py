import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

def find_flakes(image_bgr, edge_threshold=10, area_threshold=500, display=False):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    R = image_bgr[:, :, 2]
    G = image_bgr[:, :, 1]
    
    process_image = np.stack((R, G), axis=2)
    
    smoothed = cv2.GaussianBlur(process_image, (5, 5), 0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(np.sum(grad_x**2 + grad_y**2, axis=2))
    #magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    #magnitude[magnitude < threshold] = 0
    
    binary = np.where(magnitude >= edge_threshold, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    
    contour_img = image_rgb.copy()
    background_img = image_rgb.copy()
    
    cv2.drawContours(contour_img, area_filtered_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(background_img, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    
    if display:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(image_rgb)
        axs[0, 0].set_title("Original")
        
        axs[0, 1].imshow(cleaned, cmap='gray')
        axs[0, 1].set_title("Edges")
        
        axs[1, 0].imshow(contour_img)
        axs[1, 0].set_title("Detected Flakes")
        
        axs[1, 1].imshow(background_img)
        axs[1, 1].set_title("Masked Background")
        
        for ax in axs.ravel():
            ax.axis('off')
        
        #plt.suptitle(filename)
        plt.tight_layout()
        plt.show()
    
    return background_img, area_filtered_contours

if __name__ == "__main__":
    '''
    folder_path = filedialog.askdirectory(title="Select Image Folder")

    extensions = (".png", ".jpg", ".jpeg", ".bmp")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(folder_path, filename)

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

            find_flakes(image_bgr, display=True)
    '''
            
    image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    find_flakes(image_bgr, edge_threshold=5, display=True)