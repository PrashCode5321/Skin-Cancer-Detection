import cv2
import numpy as np
import os
import time

# Input Image of skin Lesion
images = ['benign/'+i for i in os.listdir('benign')]
# images += ['malignant/'+i for i in os.listdir('malignant')]
images += ['dataset/'+i for i in os.listdir('dataset')]

for image_name in images:
    img = cv2.imread(image_name)
    cv2.imshow('Input Image', img)
    original_img = img.copy()

    # Contrast Enhancement
    """
    This will shift the values of the actual image
    We might not get consistant results, will depend on the area of the lesion
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.equalizeHist(gray)
    # cv2.imshow('Contrast Enhanced Image', gray)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations = 3)
    # img = cv2.blur(closing,(10,10))
    img = cv2.GaussianBlur(closing, (3, 3), 0)

    # # Thresholding in RGB planes
    # b, g, r = cv2.split(img)
    # # cv2.imshow('Red Plane Original', r)
    # # cv2.imshow('Green Plane Original', g)
    # # cv2.imshow('Blue Plane Original', b)

    # # _, r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_OTSU, 199, 5)
    # # _, g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_OTSU, 199, 5)
    # # _, b = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_OTSU, 199, 5)
    # _, r = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
    # _, g = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    # _, b = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)
    # # cv2.imshow('Red Plane', r)
    # # cv2.imshow('Green Plane', g)
    # # cv2.imshow('Blue Plane', b)

    # # Creating binary masks using thresholding
    # mask = cv2.bitwise_and(r, g)
    # mask = cv2.bitwise_and(mask, b)
    # # cv2.imshow('Mask', mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # Find the largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) < int(0.9*img.shape[0]*img.shape[1])]
    cnt = sorted(areas, key=lambda x: x[0], reverse=True)[0][1]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(perimeter/(area**0.5))

    # Edge Detection
    new_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(new_mask, [cnt], -1, (255, 255, 255), -1)
    # cv2.imshow('Segmented Mask', new_mask)

    # Segmented image containing only the lesion
    img = cv2.bitwise_and(original_img, new_mask)
    cv2.imshow('Segmented Image', img)

    # Extracting the geometry features

    # Classification

    # Result
    close_all = False
    while True:
        if cv2.waitKey(0):
            break
        if cv2.waitKey('q'):
            close_all = True
            break
    cv2.destroyAllWindows()
    if close_all:
        break