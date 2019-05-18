#%%
"""
Created on Wed May  6 22:15:06
Main class for camera calibration

@author: cchadd

    image_p_vec: video 3D
    real_p_vec:  terrain 2D
"""

import numpy as np
import cv2
from calibration.mean_selection_calibration import MeanSelectionCalib

#%%
class Calibrator(object):

    def __init__(self, calibration_method, path_to_video, path_to_frames, name_to_frames='', num_frames=5, frame_delay=10):
        self.calibration_method = calibration_method
        self.calibrator = MeanSelectionCalib(path_to_video, path_to_frames, name_to_frames, num_frames, frame_delay)

    def calibration(self):
        self.calibrator.calibrate_camera()
        print (self.calibrator.camera_matrix)
        self.calibrator.record_testing_points()

######TO BE CHECKED#######
    def compute_projection_err(self):
        projected_points, _ = cv2.projectPoints(
           self.calibrator.image_p_vec[0],
           self.calibrator.rot_matrix[0],
           self.calibrator.tran_matrix[0],
           self.calibrator.camera_matrix,
           self.calibrator.distortion)
       
    def project_points(self, test_points):
        """
            Project new test points from video 3D to terrain 2D
        """
        projected_points = cv2.projectPoints(
           test_points,
           self.calibrator.rot_matrix[0],
           self.calibrator.tran_matrix[0],
           self.calibrator.camera_matrix,
           self.calibrator.distortion)

        return projected_points
