#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:23:59 2019
Détection automatique de ligne en Open CV:
    - Hough Lines detection

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
# lines
lines = cv2.HoughLines(edges, 1.1, np.pi/180, 150)

for i, _ in enumerate(lines):
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)
        print(rho, theta)

plt.figure(figsize=(15, 5))
plt.imshow(img)
plt.show()
