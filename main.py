import cv2

# Input Image of skin Lesion
img = cv2.imread('image.jpg').cvtColor(cv2.COLOR_BGR2RGB)

# Contrast Enhancement
img = cv2.equalizeHist(img)

# Thresholding in RGB planes
r, g, b = cv2.split(img)
_, r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Creating binary masks using thresholding

# Find the largest blob

# Edge Detection

# Segmented image containing only the lesion

# Extracting the geometry features

# Classification

# Result