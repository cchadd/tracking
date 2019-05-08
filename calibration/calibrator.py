#%%
import numpy as np
import cv2
from calibration.framesProcess import FramesProcess

path_to_video = '../video/Angle1.mp4'
path_to_store = '../test_frames/'
name_to_frames = 'test'
FramesProcess.get_frames(path_to_video, path_to_store, name_to_frames)
coord = FramesProcess.get_coordinates(path_to_store, 2)


class Calibrator(object):
    def __init__(self, path_to_frames, num_frame):

        assert isinstance(path_to_frames, str)
        assert isinstance(num_frame, int)

        self.__frames = path_to_frames
        self.__num_frame = num_frame