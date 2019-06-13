#%% Imports
import cv2
import sys
from random import randint
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#%% Set path of video
path_to_video = "/home/tdesfont/0-documents/shared_soccer/video/20123_1.mp4"
capture = cv2.VideoCapture(path_to_video)
path_to_store = './frames/'
name_to_frames = 'frame'

#%%
frame_index = -1
frame_count = 100000
frame_delay = 0
frame_period = 10

#%%
count = 0
while count < frame_count:
    frame_index += 1
    ret, frame = capture.read()
    if frame_index > frame_delay:
        if frame_index % frame_period == 0:
            count += 1
            file_name = path_to_store + name_to_frames +'_{}.jpg'.format(count)
            cv2.imwrite(file_name, frame)
capture.release()

