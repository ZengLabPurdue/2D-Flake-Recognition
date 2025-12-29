import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tkinter import filedialog

#---------------------------
# Program Parameters
#---------------------------

num_contours = 100
contour_thickness = 10
contour_color = (0,0,0) # in BGR

#----------------------------
# Load Image
#----------------------------

image_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blue_image = image[:, :, 0]
green_image = image[:, :, 1]
red_image = image[:, :, 2]

#----------------------------
# Image Processing
#----------------------------

#
# Filters
#

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)

# Apply Average blur to reduce noise
#blurred_image = cv2.blur(gray_image, (100,100))


#
# Colormap
#

# Normalize the image to 0-1 range for matplotlib
norm_image = (blurred_image- blurred_image.min()) / (blurred_image.max() - blurred_image.min())

# ['viridis', 'plasma', 'magma', 'jet']
colormap = cm.viridis

# Applying the colormap to get an RGB image
processed_image = colormap(norm_image)[..., :3]
processed_image_uint8 = (processed_image * 255).astype(np.uint8) # Convert to uint8 for openCV

#----------------------------
# Contour Processing
#----------------------------

#
# Canny Edge Detector
#
'''
canny = cv2.Canny(blurred_image, 20, 40)
'''

#
# Laplacian Edge Detector
#
'''
laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)
'''

#
# Morphological Gradient Detection
#

gray = cv2.cvtColor(processed_image_uint8, cv2.COLOR_BGR2GRAY)

# Apply threshold to get binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Create a kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# Compute morphological gradient
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# Use blurred image for thresholding
thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 3)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(f"Number of contours found: {len(contours)}")


#
# Displaying Image
#

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Gaussian Blur
plt.subplot(1, 3, 2)
plt.imshow(processed_image_uint8)
plt.title('Gaussian Blurred Image')
plt.axis('off')

# Edge-detected image
plt.subplot(1, 3, 3)
plt.imshow(gradient)
plt.title('Morphological Detection')
plt.axis('off')

plt.show()

#
# Filtering Contours
#

# Find most important contours
contourAreaList = []
for contour in contours:
    area = cv2.contourArea(contour)
    contourAreaList.append((contour, area))

sortedContourAreaList = sorted(contourAreaList, key=lambda student: student[1], reverse=True)

sortedContours = [contour[0] for contour in sortedContourAreaList]

#
# Displaying Contoured Image
#

# Drawing Contour Traces
tracedContourImage = image.copy()
cv2.drawContours(tracedContourImage, sortedContours[:num_contours], -1, contour_color, contour_thickness)
plt.imshow(cv2.cvtColor(tracedContourImage, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show() 

#------------------------------------
# Finding Width and Length of Contour
#------------------------------------

boundedContourImage = image.copy()
rect = cv2.minAreaRect(contour)
# rect = ((center_x, center_y), (width, height), angle)

distanceFromBorder = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (255, 255, 255)
thickness = 3

box = cv2.boxPoints(rect)
box = np.int32(box)

center = rect[0]
width = rect[1][0]
height = rect[1][1]
angle = rect[2]

rad_angle = math.radians(angle)
rad_angle_perp = math.radians(angle + 90)

positionWidth = (
    int(center[0] + (distanceFromBorder + height/2) * math.cos(rad_angle_perp)),
    int(center[1] + (distanceFromBorder + height/2) * math.sin(rad_angle_perp))
)

positionHeight = (
    int(center[0] + (distanceFromBorder + width/2) * math.cos(rad_angle)),
    int(center[1] + (distanceFromBorder + width/2) * math.sin(rad_angle))
)

cv2.putText(boundedContourImage, str(round(width,2)), positionWidth, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(boundedContourImage, str(round(height,2)), positionHeight, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.drawContours(boundedContourImage, [box], 0, (255, 255, 255), 5)

plt.imshow(cv2.cvtColor(boundedContourImage, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show() 

#----------------------------
# Finding Points in Contours
#----------------------------

filledContourImage = image.copy()
contour = np.array(sortedContours[0])

x, y, w, h = cv2.boundingRect(contour)
mask = np.zeros((h, w), dtype=np.uint8)
contour_shifted = contour - [x, y]
cv2.fillPoly(mask, [contour_shifted], 255)
filledContourImage[y:y+h, x:x+w][mask == 255] = (255, 255, 255)

plt.imshow(cv2.cvtColor(filledContourImage, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show() 
#points_inside_contour = list(zip(xs, ys))