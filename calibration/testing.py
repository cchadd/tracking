#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Test calibration methods and calibration class
"""

#%% Imports
from calibration.mean_selection_calibration import MeanSelectionCalib
from calibration.calibrator import Calibrator
import matplotlib.pyplot as plt
import numpy as np

#%%
path_to_video = '../video/Angle1.mp4'
path_to_store = '../test_frames/'
name_to_frames = 'test'

# TODO: Specify angle of camera
soccer_keypoint = {
        # Corners
        0: (0, 0),
        1: (0, 15),
        2: (25, 15),
        # Goal zone border
        3: (0, 4.5),
        4: (0, 10.5),
        -5: (25, 10.5),
        -6: (25, 4.5),
        # Goal cage
        -7:  (0, 6),
        -8: (0, 9),
        -2: (25, 9),
        -3: (25, 6),
        #
        -9: (3, 7.5),
        -10: (6, 7.5),
        -11: (12.5, 7.5),
        -12: (19, 7.5),
        -13: (22, 7.5)
        }

#%% Display keypoints
plt.figure(figsize=(5, 10))
for key in soccer_keypoint:
    plt.scatter(soccer_keypoint[key][0], soccer_keypoint[key][1],
                s=70, c='grey', edgecolor='k')
    coordinates = list(soccer_keypoint[key])
    plt.annotate(str(key), tuple([coordinates[0]+0.5, coordinates[1]+0.5]))

eps = 3
plt.xlim([ -eps, 25+eps])
plt.ylim([0-eps, 15+eps])
plt.plot([0, 25, 25, 0, 0], [0, 0, 15, 15, 0], 'k--')
# midline
plt.axhline(7.5, color='k')
# cages: mid+-1.5
plt.axhline(9, color='r')
plt.axhline(6, color='r')
# zones but: mid+-3
plt.axhline(10.5, color='g')
plt.axhline(4.5, color='g')

plt.show()

#%% Don't use

calib = MeanSelectionCalib(path_to_video, path_to_store, soccer_keypoint, name_to_frames, 2)
calib.calibrate_camera()
calib.camera_matrix

#%% To be used

cal = Calibrator('mean_selection', path_to_video, path_to_store, soccer_keypoint, name_to_frames,2)
cal.calibration()

test_points = np.array([[ 36.5, 72.5, 1.],
                        [192.5, 44.5, 1.],
                        [ 592., 123., 1.],
                        [  88.,  60., 1.],
                        [146.5,  49., 1.]], dtype=np.float32)

projected_points, _ = cal.project_points(test_points)

#%%
cal.compute_projection_err()