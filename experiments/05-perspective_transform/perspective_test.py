#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://www.youtube.com/watch?v=PtCQH93GucA

Created on Sat Jun  1 15:43:26 2019
@author: tdesfont
"""

import cv2
import numpy as np

#%%
absolute_path = "/home/tdesfont/0-documents/shared_soccer/video/"
file_name = "Angle1.mp4"
cap = cv2.VideoCapture(absolute_path+file_name)

while True:
    _, frame = cap.read()
    cv2.circle(frame, (72, 64), 1, (0, 0, 255), -1)
    cv2.circle(frame, (497, 333), 2, (0, 0, 255), -1)
    cv2.circle(frame, (581, 119), 3, (0, 0, 255), -1)
    cv2.circle(frame, (190, 42), 4, (0, 0, 255), -1)

    pts1 = np.float32([[72, 64], [497, 333], [581, 119], [190, 42]])
    pts2 = np.float32([[0, 1.875], [25, 1.875], [25, 15], [0, 15]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2*20)
    result = cv2.warpPerspective(frame, matrix, (500, 300))

    cv2.imshow("Frame", frame)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

#%%
absolute_path = "/home/tdesfont/0-documents/shared_soccer/video/"
file_name = "20123_1.mp4"
cap = cv2.VideoCapture(absolute_path+file_name)

while True:
    _, frame = cap.read()

    cv2.circle(frame, (109, 77), 1, (0, 0, 255), -1)
    cv2.circle(frame, (512, 336), 2, (0, 0, 255), -1)
    cv2.circle(frame, (572, 216), 3, (0, 0, 255), -1)
    cv2.circle(frame, (168, 68), 4, (0, 0, 255), -1)

    pts1 = np.float32([[109, 77], [512, 336], [168, 68], [572, 216]])
    pts2 = np.float32([[0, 0], [25, 0], [0, 15], [25, 15]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2*20)
    result = cv2.warpPerspective(frame, matrix, (500, 300))

    cv2.imshow("Frame", frame)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()