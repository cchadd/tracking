#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Test calibration methods and calibration class
"""

#%% Imports
from calibration.mean_selection_calibration import MeanSelectionCalib
from calibration.calibrator import Calibrator
from calibration.framesProcess import FramesProcess
import matplotlib.pyplot as plt
import numpy as np
import cv2
#%%
path_to_video = '../video/test.webm'
path_to_store = '../test_frames/'
name_to_frames = 'test'

# TODO: Specify angle of camera
soccer_keypoint = {
        # Corners
        0: (0, 0),
        1: (0, 10),
        2: (10, 0),
        # Goal zone border
        3: (10, 10),
        -4: (5, 5),
        -5: (5, 0),
        -6: (0, 5),
        # Goal cage
        -7:  (0, 2.5),
        -8: (0, 7.5),
        -2: (5, 2.5),
        -3: (7.5, 7.5),
        #
        -9: (0, 0),
        -10: (0, 0),
        -11: (0, 0),
        -12: (0, 0),
        -13: (0, 0)
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

cal = Calibrator('mean_selection', path_to_video, path_to_store, soccer_keypoint, name_to_frames,1)
cal.calibration()
#%%
test_points = np.array([[ 36.5,  72.5],
                        [192.5,  44.5],
                        [592. , 123.]], dtype=np.float32)

#%%
testing = cal.calibrator.real_testing_p_vec.copy()
n=testing.shape[1]
v = np.ones((n,1))
testing = np.hstack((testing[0], v))

print (n, v)
#%%
projected_points, _ = cal.project_points(cal.calibrator.real_testing_p_vec)

#%%
fig = plt.figure(figsize=(10, 10))

im = cv2.imread('../test_frames/test_1.jpg')
plt.scatter(projected_points[: , :, 0][:, 0], projected_points[:, : , 1][:, 0], s=10, color='r', alpha=0.8)
plt.imshow(im)

proj = projected_points

#%%
project = []
for element in proj:
    project.append(list(element[0]))

project = np.array(project)
#%%


n = project.shape[0]
v = np.ones((n,1))

project = np.hstack((project, v))
#%%
cv2.getAffineTransform(project[:3], cal.calibrator.real_p_vec[:, :3, :2][0])


#%%

cam = cal.calibrator.camera_matrix
rot = cv2.Rodrigues(cal.calibrator.rot_matrix[0])
tran = cal.calibrator.tran_matrix[0]
point = np.array([0, 0, 0])

mtx_inve = cv2.invert(cal.calibrator.camera_matrix)
rot_inv = cv2.invert(rot[0])
image = np.array([[5.],
                  [94.],
                  [0.]], dtype='float32')

rvec_inv = cv2.Rodrigues(rot_inv[1])
#cv2.projectPoints(image, rvec_inv[0], -tran, mtx_inve[1], None)
(rot_inv[1].dot(mtx_inve[1])).dot(image) - tran

projected_points, _ = cv2.projectPoints(
           cal.calibrator.real_testing_p_vec,
           cal.calibrator.rot_matrix[0],
           cal.calibrator.tran_matrix[0],
           cal.calibrator.camera_matrix,
           cal.calibrator.distortion)
         

fig = plt.figure(figsize=(10, 10))

im = cv2.imread('../test_frames/test_1.jpg')
plt.scatter(projected_points[: , :, 0][:, 0], projected_points[:, : , 1][:, 0], s=10, color='r', alpha=0.8)
plt.imshow(im)

proj = projected_points

#%%
img = cv2.imread('../test_frames/test_2.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cal.calibrator.camera_matrix,cal.calibrator.distortion,(w,h),1,(w,h))

undi = cv2.undistort(img, cal.calibrator.camera_matrix, cal.calibrator.distortion, None, newcameramtx)
plt.imshow(undi)
#%%
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(cal.calibrator.camera_matrix,cal.calibrator.distortion,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
plt.imshow(dst)

#%%
undi = cv2.undistort(im, cal.calibrator.camera_matrix, cal.calibrator.distortion, None, None)
plt.imshow(undi)
plt.savefig('./undistort_flux_1.png')
#%%
plt.imshow(im)

#%%
p = FramesProcess()

coord = p.get_coordinates(path_to_store, 1, 'ok')


#%%
coord

#%%
plt.imshow(im)

#%%

h,  w = im.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cal.calibrator.camera_matrix,cal.calibrator.distortion,(w,h),1,(w,h))



#%%
# undistort
dst = cv2.undistort(im, cal.calibrator.camera_matrix, cal.calibrator.distortion, None, newcameramtx)

# crop the image
x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
plt.imshow(dst)
cv2.imwrite('calibresult.png',dst)


#%%
