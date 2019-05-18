#%%
from calibration.mean_selection_calibration import MeanSelectionCalib
from calibration.calibrator import Calibrator

path_to_video = '../video/Angle1.mp4'
path_to_store = '../test_frames/'
name_to_frames = 'test'

soccer_keypoint = {
        # Corners
        0: (0, 0),
        1: (0, 15),
        2: (25, 15),
        -1: (25, 0),
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
#%%

calib = MeanSelectionCalib(path_to_video, path_to_store, soccer_keypoint, name_to_frames, 2)
calib.calibrate_camera()
calib.camera_matrix


#%%
cal = Calibrator('mean_selection', path_to_video, path_to_store, soccer_keypoint, name_to_frames,2)
cal.calibration()

#%%
cal.compute_projection_err()
