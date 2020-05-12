#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary functions

# In[1]:


import cv2
import os
import sys
import numpy
import glob
import matplotlib.pyplot as plt
import random
# ### Enhance images

# - Ridge Segmentation -> Region of interest
# - Ridge Orientation -> Every pixels orientation 
# - Ridge Frequency -> Overall frequency of ridge
# - Gabor Filter -> Ridge Filter
# 
# [Github Repo](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python)

# In[6]:


#sys.path.append('../')
from enhance import image_enhance


# ### Skeletonize
# - Morphological process for reducing foreground regions in a binary image to a skeletal remnant that largely preserves the extent & connectivity of original region, while throwing away most of the original foreground pixels.

# In[5]:


from skimage.morphology import skeletonize, thin


# ###  Ploting image

# In[7]:


def plot_figure(img1,img2,img3,a,b,c):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 10))
    ax1.imshow(img1, cmap = 'gray')
    ax2.imshow(img2, cmap = 'gray')
    ax3.imshow(img3, cmap = 'gray')

    ax1.set_title(f'{a}')
    ax1.axis('off')
    ax2.set_title(f'{b}')
    ax2.axis('off')
    ax3.set_title(f'{c}')
    ax3.axis('off')


# ### Thinning function instead of using ski image
# - Removes dots and connects lines to find corners by Harris Corner Algorithm

# In[8]:


def removedot(invertThin):
    temp0 = numpy.array(invertThin[:])
    temp0 = numpy.array(temp0)
    temp1 = temp0/255
    temp2 = numpy.array(temp1)
    temp3 = numpy.array(temp2)

    enhanced_img = numpy.array(temp0) # why?
    filter0 = numpy.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))
  
    return temp2


# 
# ### CLAHE
# 
# Contrast Limited Adaptive Histogram Equalization - To prevent global histogram equalization
# - Image is divided into small blocks openCV (8*8)
# - Each of these blocks are histogram equalized as usual.
# So in a small area, histogram would confine to a small region (unless there is noise),
# if noise is there it will be amplified.
# - To avoid this contrast limiting is applied.
# - If any Histogram bin is above the specified contrast limit (default 40) 
# Those pixels are clipped and distributed uniformly to other bins before apply histogram equalization.
# - After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.
#     

# ## Corner Harris Detection
# - Algorithm to extract the corners of an image, which can be stored as a KeyPoint.

# ### ORB Descriptor
# Oriented Fast And Rotated Brief
# - Fast Keypoint detector & Brief Descriptor
# - Fast & Accurate Orientation Component to FAST

# In[15]:

# ### Descriptor function without plots

# In[40]:


def get_descriptors(img):
    i = random.randint(0,1000) 
    # Apply Clahe
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img1 = img * 255
    a = "CLAHE" + str(i) 
    # cv2.imshow("Clahe",img)
    cv2.imwrite(f"./static/img/{a}.png",img1)
    
    # Enhance image
    img = image_enhance.image_enhance(img)
    img = numpy.array(img, dtype=numpy.uint8)
    # cv2.imshow("enhance",255*img)    
    img2 = img*255
    b = "Enhance" + str(i)
    cv2.imwrite(f"./static/img/{b}.png",img2)
    
    # Threshold
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1
    img3 = img
    c = "Normalize" + str(i)
    # cv2.imshow("Threshold",img)
    cv2.imwrite(f"./static/img/{c}.png",img3)

  #  plot_figure(img1,img2,img3,a,b,c)
    
    #Skeleton & Thinning
    skeleton = skeletonize(img)
    img1 = skeleton*255
    a = "Skeleton"+ str(i)
    cv2.imwrite(f"./static/img/{a}.png",img1)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    #cv2.imshow("skeleton",skeleton)
    skeleton = removedot(skeleton)
    img2 = skeleton*255
    b = "Thinning" + str(i)
    #cv2.imwrite(f"./static/img/{b}.png",img2)
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
            for y in range(0, harris_normalized.shape[1]):
                    if harris_normalized[x][y] > threshold_harris:
                            #print(x,y)
                            keypoints.append(cv2.KeyPoint(y, x, 1))
    img3 = cv2.drawKeypoints(img, keypoints, outImage=None)
    c="Keypoints" + str(i)
  #  plot_figure(img1,img2,img3,a,b,c)
    cv2.imwrite(f"./static/img/{c}.png",img3)

    # Define descriptor
    orb = cv2.ORB_create() 
    # Compute descriptors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des)

def get_descriptors2(img):
    # Apply Clahe
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = image_enhance.image_enhance(img)
    img = numpy.array(img, dtype=numpy.uint8)
    # Thresholding image
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Normalize to 0 and 1 range
    img[img == 255] = 1

    #Thinning
    skeleton = skeletonize(img)
    skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)
    
    # Harris corners
    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
            for y in range(0, harris_normalized.shape[1]):
                    if harris_normalized[x][y] > threshold_harris:
                           # print(x,y)
                            keypoints.append(cv2.KeyPoint(y, x, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    _, des = orb.compute(img, keypoints)
    return (keypoints, des)


# ### Brute Force match
# ### Hamming distance
# 
# [Reference](https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/)

# ### Score_Threshold - 35 below this is a match otherwise false (change Accordingly)

# In[33]:


def verify(img1,img2):
    i = random.randint(0,1000) 
    img1 = cv2.imread(img1,0)
#    print("Original Image")
    kp1,des1 = get_descriptors(img1)
    img2 = cv2.imread(img2,0)
#    print("Candidate Image")
    kp2, des2 = get_descriptors(img2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
   # Plot keypoints
#    print("plotting on original images and match")
    img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
    a = "Original Keypoints" + str(i)
    cv2.imwrite(f"./static/img/{a}.png",img4)

    img5 = cv2.drawKeypoints(img2, kp2, outImage=None)
    b = "Candidate Keypoints" + str(i)
    cv2.imwrite(f"./static/img/{b}.png",img5)

    img6 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    c = "Match" + str(i)
    cv2.imwrite(f"./static/img/{c}.png",img6)

 #   plot_figure(img4,img5,img6,a,b,c)
    
    
    score = 0
    for match in matches:
            score += match.distance
    score_threshold = 36
    s = (score/len(matches) )
    if score/len(matches) < score_threshold:
        return (f"Fingerprint matches with Avg. Difference Score = {s} when Threshold = {score_threshold}")
    else:
        return(f"Fingerprint does not match with Avg. Difference Score = {s} when Threshold = {score_threshold}")


# ### 1:1 Match

# In[36]:


#img1 = "../Devashish/Normal/Right Thumb_158793710430.png"
#img2 = "../Devashish/rotated/Right Thumb_158793731879.png"
#verify(img1,img2)




