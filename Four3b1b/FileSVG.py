import numpy as np
import cv2
# from PIL import Image
"""
 np.exp^360"""

img_path = ''

# using canny filter to get edges
img = cv2.imread(img_path)

np_img = np.array(img)
