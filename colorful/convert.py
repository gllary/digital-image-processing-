# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:42:55 2019

@author: 官宇霞
"""

import os.path

import cv2

img_fold_A = 'E:/QQ/test1/before/b'

img_list1 = os.listdir(img_fold_A)

num_imgs1 = len(img_list1)

for i in range(num_imgs1):
    save_fold_A = 'E:/QQ/test1/before/a'
    name_A = img_list1[i]
    path_A = os.path.join(img_fold_A, name_A)
    im_A = cv2.imread(path_A, 1)
    file_name_temp = name_A[:-4]
    file_name = os.path.join(save_fold_A , file_name_temp+'.jpg')
    cv2.imwrite(file_name, im_A)