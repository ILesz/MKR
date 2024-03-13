import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path=#NorthBroadSt_Landscape_1_M.Edlow.jpg"
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap="gray")
plt.show()

img_edges = cv2.Canny(img_gray, threshold1=100, threshold2=100)
plt.imshow(img_edges, cmap="gray")
plt.show()

img_red_min_green = img[:, :, 0]-img[:, :, 1]
plt.imshow(img_red_min_green, cmap="gray")
plt.show()

corners = cv2.cornerHarris(np.float32(img_gray), 2, 5, 0.07)
corners = cv2.dilate(corners, None)
img_cp = img.copy()
img_cp[corners > 0.01 * corners.max()]=[0, 0, 255]
plt.imshow(img_cp)
plt.show()

corners[corners > 0.01 * corners.max()] = 255
corners[corners!=255] = 0
plt.imshow(corners, cmap="gray")
plt.show()

rg_contrast = img_red_min_green.copy()
rg_contrast[rg_contrast>150] = 255
rg_contrast[rg_contrast!=255] = 0
rg_edges = cv2.Canny(img_red_min_green, threshold1=100, threshold2=100)
plt.imshow(rg_edges, cmap="gray")
plt.show()

rg_corners = cv2.cornerHarris(np.float32(rg_contrast), 2, 5, 0.07)
rg_corners = cv2.dilate(rg_corners, None)
rg_corners[rg_corners > 0.01 * rg_corners.max()] = 255
rg_corners[rg_corners!=255] = 0
plt.imshow(rg_corners, cmap="gray")
plt.show()
