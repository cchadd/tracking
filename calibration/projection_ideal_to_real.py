#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Calibration and projection matrix
"""

#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#%% Slice videos from both angles into frames
for angle_index in [1, 2]:
    angle_name = 'Angle{}'.format(angle_index)

    cap = cv2.VideoCapture('../video/{}.mp4'.format(angle_name))
    frame_delay = 10
    frame_count = 5
    count = 0
    i = 0
    while count < frame_count:
        i += 1
        ret, frame = cap.read()
        if i % frame_delay == 0:
            count += 1
            file_name = './frames/{}_frame_{}.jpg'.format(angle_name, count)
            print('Created frame {}'.format(count))
            cv2.imwrite(file_name, frame)

    cap.release()
#%% Process one frame and copy
img_right = cv2.imread('./frames/Angle1_frame_1.jpg')
im = img_right.copy()
#%% Function for interest points selection

cv2.namedWindow('image')

crd = []
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, crd
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        crd.append(refPt[0])
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(im, refPt[0], refPt[1], (0, 0, 255), 2)
        cv2.imshow("image", im)

def get_obj_point(path, num_frame, angle):
    '''
    Returns coordinates of selected points
    Inputs:
    -------
    path (str):
        String to folder with calibration frames
    num_frame (int):
        number of frames used to perform calibration
    '''
    assert isinstance(path, str)
    assert isinstance(angle, str)
    assert isinstance(num_frame, int)

    global crd, im
    coord = []
    count = 0

    # Frames selection
    regex = 'Angle{}_frame_{}.jpg'

    if angle == 'right':
        angle_index = 1
    else:
        angle_index = 2
    frames = [regex.format(angle_index, i+1) for i in range(num_frame)]


    for frame in frames:
        img_right = cv2.imread(path + frame)
        im = img_right.copy()
        while True:
            cv2.setMouseCallback("image", click_and_crop)
            cv2.imshow("image", im)
            key = cv2.waitKey(1) & 0xFF
            cv2.imwrite('calibrated.jpg', im)
            if key == ord('c'):
                count += 1
                coord.append(crd)
                crd = []
                break
    cv2.destroyAllWindows()
    return coord

#%% Build dictionary points for the planar soccer field

soccer_keypoint = {
        # Corners
        0: (0, 0),
        1: (0, 15),
        2: (25, 15),
        -1: (25, 0),
        # Goal zone border
        3: (0, 4.5),
        4: (0, 10.5),
        5: (25, 10.5),
        6: (25, 4.5),
        # Goal cage
        7:  (0, 6),
        8: (0, 9),
        -2: (25, 9),
        -3: (25, 6),
        #
        9: (3, 7.5),
        10: (6, 7.5),
        11: (12.5, 7.5),
        12: (19, 7.5),
        13: (22, 7.5)
        }

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
#
plt.show()

#%%

path_to_frames = './frames/'
num_frame = 1
angle = 'right'

right_points = get_obj_point(path_to_frames, num_frame, angle)


#%%
image_points = []
for frame_num in range (0, len(right_points)):
    frame_num_points = []
    for point in right_points[frame_num]:
        frame_num_points.append(list(point))
    image_points.append(np.array(frame_num_points))

#%%

# Store the coordinates in a matrix
# np.save('image_points', image_points)

# image_points = np.load('image_points.npy')

#%% Display the chosen points
interest_points_vec = []
for point in right_points[0]:
    interest_points_vec.append(list(point))
interest_points_vec = np.array(interest_points_vec)

plt.figure(figsize=(15, 5))
plt.imshow(im, cmap='jet')
plt.scatter(interest_points_vec[:, 0], interest_points_vec[:, 1], s=60,
            alpha=0.5, edgecolor='k', color='w')
for i, point in enumerate(interest_points_vec):
    plt.annotate(str(i), tuple(point))
plt.show()

#%%
object_points = []
keys = [i for i in sorted(list(soccer_keypoint.keys())) if i>=0]
for key in keys:
    object_points.append(list(soccer_keypoint[key]))
object_points = np.array(object_points)

#%%

image_points = image_points[0]

try:
    assert image_points.shape == object_points.shape
    print('Success ! Same number of image and object points...')
except:
    print('Dimension of image and object points not matching.')
    print(image_points.shape, object_points.shape)

#%%Calibration

object_p_vec = object_points.copy()
image_p_vec = image_points.copy()

key_selection = [2, 5, 12, 13, 6]
object_p_vec = object_p_vec[key_selection]
image_p_vec = image_p_vec[key_selection]

n = object_p_vec.shape[0]
v = np.ones ((n, 1))

object_p_vec = np.hstack((object_p_vec, v))
object_p_vec = object_p_vec.reshape(1, -1, 3).astype('float32')

image_p_vec = image_p_vec.reshape(1, -1, 2).astype('float32')

#%% Calcul de la matrice d'homotopie
camera_matrix = cv2.initCameraMatrix2D([object_p_vec], [image_p_vec],
                                       im.shape[:2])

ret, mtx, dist, rvecs, tvecs, = cv2.calibrateCamera(
        [object_p_vec], [image_p_vec],
        im.shape[:2], camera_matrix,
        None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

#%% From object to image

i = 0 # Since there is only one temporal frame
projected_points, _ = cv2.projectPoints(object_p_vec[i], rvecs[i], tvecs[i],
                                        mtx, dist)

# Display reference points
plt.figure()
plt.imshow(im)
plt.scatter(projected_points[:, 0][:, 0], projected_points[:, 0][:, 1])
plt.show()

#%%
n = 10
plt.figure()
plt.imshow(im)

for y_coord in np.linspace(7, 12, 3):
    grid_points = np.empty((n, 3))
    grid_points[:, 0] = np.linspace(0, 25, n)
    grid_points[:, 1] = np.ones(n)*y_coord
    projected_points, _ = cv2.projectPoints(grid_points, rvecs[i], tvecs[i],
                                            mtx, dist)
    plt.plot(projected_points[:, 0][:, 0], projected_points[:, 0][:, 1], 'r-')
plt.ylim(340, 0)
plt.xlim(0, 640)
plt.show()


