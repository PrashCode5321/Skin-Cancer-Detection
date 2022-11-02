import cv2
import numpy as np
import os
import time

#INPUT IMAGES
images = ['benign/'+i for i in os.listdir('benign')]
#images += ['malignant/'+i for i in os.listdir('malignant')]
images += ['malignant/'+i for i in os.listdir('malignant')]

def CLAHE(img : np.ndarray) -> np.ndarray:
    '''
    Performs CLAHE enhancement of the image
    '''
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # clahe Limit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    img = clahe.apply(image_bw) + 30

    return img

def CLAHE2(img : np.ndarray) -> np.ndarray:
    '''
    Performs CLAHE enhancement on images of LAB color space
    '''
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

def KMeans(image : np.ndarray, k : int, mask : bool) -> np.ndarray:
    '''
    Performs K-means segmentation of the image
    '''
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,centers = cv2.kmeans(image, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    final_img = centers[label.flatten()]
    final_img = final_img.reshape(image.shape)
    
    if mask:
        k = 1
        for center in centers:
        # select color and create mask
        #print(center)
            layer = final_img.copy()
            mask = cv2.inRange(layer, center, center)
        # apply mask to layer 
        layer[mask == 0] = [0,0,0]
        return layer
    else:
        return final_img

def morphImg(image : np.ndarray) -> np.ndarray:
    '''
    Applies preset morphological operations on the image
    '''
    img = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    gray = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
    cv2.imshow('Noisy Image', img)
   
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    #kernel = np.ones((3,3),np.uint8)
    #kernel = np.array([1,1,1], dtype = np.uint8)
    #kernel = np.array([[1, 1, 1],[1, 1, 1]], dtype = np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1]], dtype = np.uint8)
    #kernel = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype = np.uint8)
    morph = img.copy()

    morph = cv2.dilate(morph, kernel, iterations=2)
    morph = cv2.erode(morph, kernel, iterations=4)
   
    for i in range(3):
        
        morph = cv2.morphologyEx(morph,cv2.MORPH_CLOSE,kernel, iterations = 1)
        morph = cv2.erode(morph, kernel, iterations=1)
        morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN,kernel, iterations = 1)
    morph = cv2.dilate(morph, kernel, iterations=1)
        
        
    '''kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    morph = cv2.filter2D(morph, -1, kernel)'''
    
    return morph

def img_prc(image : np.ndarray) -> np.ndarray:
    '''
    Performs image preprocessing for skin cancer detection
    '''
    #CONTRAST ENHANCEMENT
    #enh_img = CLAHE(img = image)
    enh_img = CLAHE2(img = image)
    cv2.imshow('Enhanced Image', enh_img)
    
    #MORPHOLOGICAL OPERATIONS AND SEGMENTATION
    moprh_img = morphImg(enh_img)
    cv2.imshow('Morphed Image', moprh_img)
    
    #SEGMENTATION
    #mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    #mask = KMeans(image = closing, k = 3, mask = True)
    
    img = cv2.cvtColor(moprh_img, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #FEATURE EXTRACTION
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) < int(0.9*img.shape[0]*img.shape[1])]
    cnt = sorted(areas, key=lambda x: x[0], reverse=True)[0][1]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(perimeter/(area**0.5))

    #EDGE DETECTION
    new_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(new_mask, [cnt], -1, (255, 255, 255), -1)
    cv2.imshow('Segmented Mask', new_mask)

    #SEGMENTING THE FEATURES
    img = cv2.bitwise_and(enh_img, new_mask)

    return img 

for image_name in images:
    img = cv2.imread(image_name)
    img = cv2.resize(img, (400, 400))
    cv2.imshow('Input Image', img)
    
    img = img_prc(image = img)
    
    cv2.imshow('Segmented Image', img)

    pr_name = "pr_" + image_name
    cv2.imwrite(pr_name, img)

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