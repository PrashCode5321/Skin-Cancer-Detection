import cv2
import numpy as np

# Input Image of skin Lesion
img = cv2.imread('dataset/ISIC_0024308.jpg')
cv2.imshow('Input Image', img)

# Contrast Enhancement
"""
This will shift the values of the actual image
We might not get consistant results, will depend on the area of the lesion
"""
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray = cv2.equalizeHist(gray)
# cv2.imshow('Contrast Enhanced Image', gray)

# Thresholding in RGB planes
b, g, r = cv2.split(img)
# cv2.imshow('Red Plane Original', r)
# cv2.imshow('Green Plane Original', g)
# cv2.imshow('Blue Plane Original', b)

_, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('Red Plane', r)
# cv2.imshow('Green Plane', g)
# cv2.imshow('Blue Plane', b)

# Creating binary masks using thresholding
mask = cv2.bitwise_and(r, g)
mask = cv2.bitwise_and(mask, b)
cv2.imshow('Mask', mask)

# Find the largest blob
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) < int(0.9*img.shape[0]*img.shape[1])]
cnt = sorted(areas, key=lambda x: x[0], reverse=True)[0][1]

# Edge Detection
new_mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(new_mask, [cnt], -1, (255, 255, 255), -1)
cv2.imshow('Segmented Mask', new_mask)

# Segmented image containing only the lesion
img = cv2.bitwise_and(img, new_mask)
cv2.imshow('Segmented Image', img)

# Extracting the geometry features

# Classification

# Result

cv2.waitKey(0)
cv2.destroyAllWindows()