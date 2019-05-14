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
                                          varThreshold=100)

cap = cv2.VideoCapture(absolute_path+file_name)

count = 0
while(cap.isOpened()):

    ret, frame = cap.read()
    roi_frame = cv2.bitwise_and(frame, frame, mask = mask.astype('int8'))
    fgmask = fgbg.apply(roi_frame)
    roi_motion = cv2.bitwise_and(roi_frame, roi_frame, mask = fgmask.astype('int8'))

    cv2.imshow('frame_1', roi_motion)
    cv2.imshow('frame_2', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

