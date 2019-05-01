#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:23:59 2019
Détection automatique de ligne en Open CV:
    - Hough Lines Probabilistic

@author: tdesfont
"""

#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%% Read images
path_image = '../../frames/'
image_name = 'Angle1_frame_1.jpg'
# read image
img = cv2.imread(path_image + image_name, 0)

#%% Apply Probabilistic Hough Lines
# canny filter (edge detection)
edges = cv2.Canny(img, threshold1=25, threshold2=100, apertureSize=3)
# lines
minLineLength = 50
maxLineGap = 12
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

for i in range(lines.shape[0]):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2), (0,255, 255),2)

plt.imshow(img)