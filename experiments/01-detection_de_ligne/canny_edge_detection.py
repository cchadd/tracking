#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:23:59 2019
Détection automatique de ligne en Open CV:
    - Canny edge detection

@author: tdesfont
"""

#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
#%% Read images

path_image = '../../frames/'
image_name = 'Angle1_frame_1.jpg'
# read image
img = cv2.imread(path_image + image_name, 0)
# canny filter (edge detection)
edges = cv2.Canny(img, threshold1=100, threshold2=200)
#%%
plt.figure(figsize=(15, 5))
#plt.imshow(img, cmap = 'gray')
plt.imshow(edges, cmap = 'viridis', alpha=0.5)
plt.show()
#%%

t1_range = range(50, 150, 20)
t2_range = range(100, 200, 20)

n = len(t1_range)

plt.figure(figsize=(15, 15))
for i, threshold1 in enumerate(t1_range):
    for j, threshold2 in enumerate(t2_range):
        plt.subplot(n, n, i*n+j+1)
        plt.title('thr1:{}, thr2:{}'.format(threshold1, threshold2))
        edges = cv2.Canny(img, threshold1, threshold2)
        plt.imshow(edges, cmap = 'magma', alpha=0.9)
plt.show()