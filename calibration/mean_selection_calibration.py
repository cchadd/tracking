import numpy as np 
import cv2
from calibration.framesProcess import FramesProcess


class MeanSelectionCalib(FramesProcess):

    def __init__(self, path_to_video, path_to_frames, real_coordinates, name_to_frames='', num_frames=5, frame_delay=10):
        
        assert isinstance(num_frames, int)
        assert isinstance(real_coordinates, dict)

        FramesProcess.__init__(self)
        self.__path_to_video = path_to_video
        self.__path_to_frames = path_to_frames
        self.__name_to_frames = name_to_frames
        self.__num_frames = num_frames
        self.__frame_delay = frame_delay
        self.__image_coordinates = []
        self.__real_coordinates = real_coordinates
        
        
    def calibrate_camera(self):
        self.get_frames(
                    self.__path_to_video,
                    self.__path_to_frames,
                    self.__name_to_frames,
                    self.__frame_delay,
                    self.__num_frames)

        image_coordinate = self.get_coordinates(
                                                    self.__path_to_frames,
                                                    self.__num_frames)

        for frame in image_coordinate:
            for point in frame:
                self.__image_coordinates.append(np.array(list(point)))

##### TO BE CONTINUED ######


