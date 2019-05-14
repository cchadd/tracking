#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:38:24 2019
@author: tdesfont
    A basic movement detection using OpenCV and the ROI mask.
"""

#%% Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%%
absolute_path = "./video/"
file_name = "Angle2.mp4"

#%% Import mask
mask = np.load("./shared/ball_detection/mask.npy")
if False:
    plt.figure()
    plt.imshow(mask)
    plt.show()

#%% Read a video with ROI mask

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,
                                          history=10,
                                          varThreshold=30)

cap = cv2.VideoCapture(absolute_path+file_name)

x = np.arange(-0.2, 0.2, 0.09)
y = np.arange(-0.2, 0.2, 0.09)
xx, yy = np.meshgrid(x, y, sparse=True)
z = 1/(xx**2 + yy**2)
z /= np.sum(z)
h = plt.contourf(x,y,z)
plt.show()

kernel = z

#kernel_size = 5
#kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
#kernel /= (kernel_size * kernel_size)
ddepth = -1

count = 0
while(cap.isOpened()):

    ret, frame = cap.read()
    roi_frame = cv2.bitwise_and(frame, frame, mask = mask.astype('int8'))
    fgmask = fgbg.apply(roi_frame)
    roi_motion = cv2.bitwise_and(roi_frame, roi_frame, mask = fgmask.astype('int8'))

    cv2.imshow('roi_motion', roi_motion)
    cv2.imshow('frame', frame)

    dst = cv2.filter2D(roi_motion, ddepth, kernel)
    cv2.imshow('filtered_motion', dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

