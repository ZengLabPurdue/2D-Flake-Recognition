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
