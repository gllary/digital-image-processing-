# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:36:17 2019

@author: 官宇霞
"""

# coding=utf-8
import time
time1 = time.time()
import cv2
import os.path
image=cv2.imread("E:/QQ/test1/after/lady.jpg")
res = cv2.resize(image, (394,1024), interpolation=cv2.INTER_AREA)
# cv2.imshow('image', image)
# cv2.imshow('resize', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("E:/QQ/test1/after/lady2.jpg",res)
time2=time.time()
print (u'总共耗时：' + str(time2 - time1) + 's')
