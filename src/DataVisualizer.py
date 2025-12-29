import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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


#----------------------------
# 3D Surface Graphing
#----------------------------

'''
Z = blurred_image.squeeze()

# Create coordinate grid
Z_small = Z[::10, ::10]

x = np.arange(Z_small.shape[1])
y = np.arange(Z_small.shape[0])
X, Y = np.meshgrid(x, y)

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z_small, linewidth=0, antialiased=False)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z value")

plt.show()
'''

#----------------------------
# 3D Visualization of Channel Data
#----------------------------

pixels = image.reshape(-1, 3)
pixels = pixels[::100] 
pixels = np.unique(pixels, axis=0) 

r = pixels[:, 0]
g = pixels[:, 1]
b = pixels[:, 2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(r, g, b, c=pixels/255.0, s=2)

ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")

plt.show()