# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:43:19 2019

@author: 官宇霞
"""
import os.path
import cv2
import os
import numpy as np
MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.15
def covert_zip():
    img_fold_A = 'E:/QQ/test1/before'
    img_list1 = os.listdir(img_fold_A)
    num_imgs1 = len(img_list1)
    for i in range(num_imgs1):
        save_fold_A = 'E:/QQ/test1/after'
        name_A = img_list1[i]
        path_A = os.path.join(img_fold_A, name_A)
        im_A = cv2.imread(path_A, 1)
        if('.tif' in name_A):
            file_name_temp = name_A[:-4]
            file_name = os.path.join(save_fold_A , file_name_temp+'.jpg')
            cv2.imwrite(file_name, im_A)
            image = cv2.imread(file_name)
            res = cv2.resize(image, (394,1024), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_name,res)
        else:
            file_name_temp = name_A[:-4]
            file_name = os.path.join(save_fold_A , file_name_temp+'.jpg')
            cv2.imwrite(file_name, im_A)
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad
def alignImages(im1,im2):
 
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
  #print(descriptors1.shape)
  #print(descriptors2.shape)
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
 
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
 
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
    
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  
  height, width = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  return im1Reg
def align(cropImg1,cropImg2):
    corners1 = cv2.goodFeaturesToTrack(cropImg1,25,0.01,10)
    m = int(len(corners1))
    corners1 = corners1.reshape(m, 2)
    corners2 = cv2.goodFeaturesToTrack(cropImg2,25,0.01,10)
    n = int(len(corners2))
    corners2 = corners2.reshape(n, 2)
 
    points1 = corners1
    points2 = corners2
    sumX1 = 0
    sumY1 = 0
    x,y = points1.shape
    for m in range(x):
        sumX1 = sumX1 + points1[m][0]
        sumY1 = sumY1 + points2[m][1]
    sumX2 = 0
    sumY2 = 0
    x,y = points2.shape
    for n in range(x):
        sumX2 = sumX2 + points2[n][0]
        sumY2 = sumY2 + points2[n][1]
    c1 = int((sumX2 / n) - (sumX1 / m))
    c2 = int((sumY2 / n) - (sumY1 / m))
    cropImg1=np.roll(cropImg1,shift=c1,axis=0)
    cropImg1=np.roll(cropImg1,shift=c2,axis=1)
    return cropImg1

def clear(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gradX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    img_gradient = cv2.subtract(gradX, gradY)
    img_gradient = cv2.convertScaleAbs(img_gradient)
    blurred = cv2.blur(img_gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    closed = np.clip(closed, 0, 255)
    closed = np.array(closed,np.uint8)
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    height,width,length = img.shape
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    x1_ = (x2-x1)/2-140
    x2_ = (x2-x1)/2+140
    if x1_< 0:
        x1_ = 0
    if x2_ > width:
        x2_ = width
    y1 = min(Ys)
    y2 = max(Ys)
    y1_ = (y2 - y1)/2-140
    y2_ = (y2 - y1)/2+140
    if y1_ < 0:
        y1_ = 0
    if y2_ > height:
        y2_ = height
    img2 = img[int(y1_):int(y2_), int(x1_):int(x2_),:]
    return img2

if __name__ == '__main__':
    
    covert_zip()
    # Read 8-bit color image.
    # This is an image in which the three channels are
    # concatenated vertically.
    for root, dirs, files in os.walk('E:/QQ/test1/after/'):  #这里路径需要修改！
          for file in files:
              print(file)
              #讀入圖像
              img_path = 'E:/QQ/test1/after/'+file    #这里路径需要修改！
              im = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
              # Find the width and height of the color image
              sz=im.shape
              height = int(sz[0] / 3);
              width = sz[1]
              print(sz)
              # Extract the three channels from the gray scale image
              # and merge the three channels into one color image
              im_color = np.zeros((height,width,3), dtype=np.uint8 )
              for i in range(0,3) :
                  im_color[:,:,i] = im[ i * height:(i+1) * height,:]
              # Allocate space for aligned image

              # The blue and green channels will be aligned to the red channel.
              # So copy the red channel

              # Define motion model
              warp_mode = cv2.MOTION_HOMOGRAPHY

              # Set the warp matrix to identity.
              if warp_mode == cv2.MOTION_HOMOGRAPHY :
                  warp_matrix = np.eye(3, 3, dtype=np.float32)
              else :
                  warp_matrix = np.eye(2, 3, dtype=np.float32)

              # Set the stopping criteria for the algorithm.
              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

              # Warp the blue and green channels to the red channel
              for i in range(0,2) :
                  im_color[:,:,i]=alignImages(im_color[:,:,i],im_color[:,:,2])
              # Show final output
              if '.jpg' in file:
                  im_color=clear(im_color)
              cv2.imwrite('E:/QQ/test1/result/a/'+file,im_color)  #这里路径需要修改！

