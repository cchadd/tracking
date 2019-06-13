#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Apply grid on the new video.
"""

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
path_to_store = './'
name_to_frames = 'frame'

#%%
frame_index = -1
frame_count = 10
frame_delay = 10000
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

#%%
# Try to find the 4th corner
window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('frame_1.jpg', cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [0, 0, 0]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
cv.imshow(window_name, dst)

# Set it to [310, 605]

# Approximately set it to [346, 640]
points = np.array([[90, 103],
                   [243, 80],
                   [621, 197],
                   [343, 650]])

img = np.copy(dst)
for i in range(4):
    img = cv.circle(img, tuple(points[i]), 3, (255,0,0), -1)

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm:
        return vec / norm
    else:
        # vec is null
        return vec

class line:

    def __init__(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)
        self.unit = normalize(self.A - self.B)
        self.normal = normalize(np.array([-self.unit[1], self.unit[0]]))

    def __mul__(self, l2):
        """
            Get intersection point of the two lines
        """
        scalar_prod = np.sum(self.unit*l2.normal)
        if not scalar_prod:
            print("Lines are parallel, no secant point.")
            return None
        else:
            M = self.B + self.unit*np.sum((l2.B-self.B)*(l2.normal))/np.sum((self.unit*l2.normal))
        return M

grid = set()
Ia_global = None
Ib_global = None

class parallelogram:

    def __init__(self, points):
        """
            ordered points
        """
        global grid
        self.points = points
        self.center = line(self.points[0], points[2])*line(self.points[1], points[3])
        grid.add((self.center[0], self.center[1]))
        global img
        img = cv.circle(img, (int(self.center[0]), int(self.center[1])), 3, (255,0,0), -1)
        for point in self.points:
            grid.add((point[0], point[1]))
            img = cv.circle(img, (int(point[0]), int(point[1])), 3, (255,0,0), -1)

    def divide(self):
        global grid
        global Ia_global
        global Ib_global
        #
        Ia = line(self.points[0], self.points[3])*line(self.points[1], self.points[2])
        A_ = line(Ia, self.center)*line(self.points[0], self.points[1])
        C_ = line(Ia, self.center)*line(self.points[2], self.points[3])
        grid.add((A_[0], A_[1]))
        grid.add((C_[0], C_[1]))
        Ia_global = Ia

        #
        Ib = line(self.points[0], self.points[1])*line(self.points[2], self.points[3])
        B_ = line(Ib, self.center)*line(self.points[1], self.points[2])
        D_ = line(Ib, self.center)*line(self.points[0], self.points[3])
        grid.add((B_[0], B_[1]))
        grid.add((D_[0], D_[1]))
        Ib_global = Ib

        # new parallelograms
        p1 = [self.points[0], A_, self.center, D_]
        p2 = [self.points[1], B_, self.center, A_]
        p3 = [self.points[2], C_, self.center, B_]
        p4 = [self.points[3], D_, self.center, C_]

        global img
        img = cv.circle(img, (int(A_[0]), int(A_[1])), 3, (255,0,0), -1)
        img = cv.circle(img, (int(B_[0]), int(B_[1])), 3, (255,0,0), -1)
        img = cv.circle(img, (int(C_[0]), int(C_[1])), 3, (255,0,0), -1)
        img = cv.circle(img, (int(D_[0]), int(D_[1])), 3, (255,0,0), -1)

        return p1, p2, p3, p4

def plug(parallelogram):
    points_ = np.empty((0, 2))
    for point in parallelogram:
        points_ = np.vstack((points_, point))
    return points_

def recursive_ortho_grid(n, points):
    if n > 0:
        P = parallelogram(points)
        sub_parallelo = P.divide()
        Ps_new = [plug(p) for p in  sub_parallelo]
        for p in Ps_new:
            recursive_ortho_grid(n-1, p)
    else:
        pass

n_pass = 4
recursive_ortho_grid(n_pass, points)
cv.imshow(window_name, img)

#%% Retrieve grid
expected_npoints = (1+ 2**n_pass)**2
X = np.empty((0, 2))
for point in grid:
    X = np.vstack((X,np.array([point[0], point[1]])))
print('Numeric error on {} points.'.format(len(X)-expected_npoints))

#%% Remove close points up to a precision of 1e-6
Y = np.empty((0, 2))
for point_out in X:
    is_close = False
    for point_in in Y:
        if np.allclose(point_in, point_out):
            is_close = True
    if not is_close:
        Y = np.vstack((Y, point_out))

assert len(Y)==expected_npoints

#%%
plt.figure(figsize=(15, 15))
plt.plot([Ia_global[0]], [Ia_global[1]], 'ro')
plt.plot([Ib_global[0]], [Ib_global[1]], 'ro')
plt.plot(Y[:, 0], Y[:, 1], 'x')
plt.show()

#%% Build the grid
n = len(Y)
vec = (Ia_global - Y)
norm_vec = 1/np.linalg.norm(Ia_global - Y, axis=1)
norm_vec = norm_vec.reshape((n, 1))
unit_vec = np.multiply(vec, norm_vec)

angle = []
for unit_vec_ in unit_vec:
    angle.append(np.angle(unit_vec_[0]+unit_vec_[1]*1j))
angle_a = angle

angle_vec = np.array(angle)
is_visited = np.zeros(len(angle_vec))
dict_key = 0
import collections
L = collections.defaultdict(list)
while 0 in is_visited:
    angle_val = angle_vec[np.where(is_visited==0)[0][0]]
    dist = np.abs(angle_val - angle_vec)
    in_line = np.where(dist<1e-5)[0]
    is_visited[in_line] = 1
    for index_ in in_line:
        L[dict_key].append(index_)
    dict_key += 1
for i in L:
    if len(L[i])!=17:
        print('That\'s a fail')

#%%
n = len(Y)
vec = (Ib_global - Y)
norm_vec = 1/np.linalg.norm(Ib_global - Y, axis=1)
norm_vec = norm_vec.reshape((n, 1))
unit_vec = np.multiply(vec, norm_vec)

angle = []
for unit_vec_ in unit_vec:
    angle.append(np.angle(unit_vec_[0]+unit_vec_[1]*1j))
angle_b = angle

angle_vec = np.array(angle)
is_visited = np.zeros(len(angle_vec))
dict_key = 0
import collections
C = collections.defaultdict(list)
while 0 in is_visited:
    angle_val = angle_vec[np.where(is_visited==0)[0][0]]
    dist = np.abs(angle_val - angle_vec)
    in_line = np.where(dist<1e-5)[0]
    is_visited[in_line] = 1
    for index_ in in_line:
        C[dict_key].append(index_)
    dict_key += 1
for i in C:
    if len(C[i])!=17:
        print('That\'s a fail')

#%%
plt.figure(figsize=(15, 15))
plt.gca().invert_yaxis()
column_end_points = np.empty((0, 2))
for i in C:
    points = Y[C[i]]
    index = list(range(len(points)))
    index = sorted(index, key=lambda i: points[i][0])
    points = points[index]
    plt.plot(points[:, 0], points[:, 1], '-')
    plt.annotate('C{}'.format(i),
                 xy=(points[0][0], points[0][1]),
                 xytext=(points[0][0], points[0][1]))
    column_end_points = np.vstack((column_end_points, np.array([points[0][0], points[0][1]])))
column_mapping = sorted(list(range(len(column_end_points))),
                        key=lambda i: column_end_points[i][0])

line_end_points = np.empty((0, 2))
for i in L:
    points = Y[L[i]]
    index = list(range(len(points)))
    index = sorted(index, key=lambda i: points[i][0])
    points = points[index]
    plt.plot(points[:, 0], points[:, 1], '-')
    plt.annotate('L{}'.format(i),
                 xy=(points[0][0], points[0][1]),
                 xytext=(points[0][0], points[0][1]))
    line_end_points = np.vstack((line_end_points, np.array([points[0][0], points[0][1]])))
line_mapping = sorted(list(range(len(line_end_points))),
                        key=lambda i: line_end_points[i][0])
plt.show()

#%%
couple = []
for a, b in zip(angle_a, angle_b):
    couple.append((round(a, 5), round(b, 5)))
y_sorted_index = list(range(len(Y)))
Y_sorted = Y[sorted(y_sorted_index, key=lambda i: couple[i])]
plt.figure(figsize=(15, 15))
for index, y in enumerate(Y_sorted):
    plt.scatter([y[0]], [y[1]], alpha=0.5, color='red', edgecolor='k')
    plt.annotate(str(index),
                 xy=(y[0], y[1]),
                 xytext=(y[0], y[1]))
plt.show()
#%% Création d'un dictionnaire
n = (1+ 2**n_pass)
coord_to_point = {}
for index, y in enumerate(Y_sorted):
    key = (index//n, index%n)
    coord_to_point[key] = y

plt.figure(figsize=(15, 15))
for coord in coord_to_point:
    point = coord_to_point[coord]
    plt.scatter([point[0]], [point[1]], alpha=0.5, color='red', edgecolor='k')
    plt.annotate(str(coord), xy=(point[0], point[1]), xytext=(point[0], point[1]))
plt.show()

#%%

window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('frame_1.jpg', cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [255, 255, 255]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
img = np.copy(dst)

plt.figure(figsize=(15, 15))
plt.imshow(img)
for coord in coord_to_point:
    point = coord_to_point[coord]
    plt.scatter([point[0]], [point[1]], alpha=0.5, color='red', edgecolor='k')
    plt.annotate(str(coord), xy=(point[0], point[1]), xytext=(point[0], point[1]))
plt.show()

#%%

# given any point on the image
sample_point = np.array([133, 106])

def discrete_projection_on_ideal(sample_point):
    global coord_to_point
    closest_coord = None
    min_dist = np.inf
    for coord in coord_to_point:
        grid_point = coord_to_point[coord]
        dist = np.linalg.norm(grid_point - sample_point)
        if dist < min_dist:
            min_dist = dist
            closest_coord = coord
    return closest_coord

discrete_projection_on_ideal(sample_point)

#%%
plt.figure(figsize=(15, 15))
plt.imshow(img)
for coord in coord_to_point:
    point = coord_to_point[coord]
    coord = discrete_projection_on_ideal(point)
    coord = (16-coord[1], coord[0])
    coord = (coord[0]*(15/16), coord[1]*(25/16))
    plt.scatter([point[0]], [point[1]], alpha=0.5, color='red', edgecolor='k')
    plt.annotate(str(coord), xy=(point[0], point[1]), xytext=(point[0], point[1]))
plt.show()

#%% Display point function

def coord_ideal(point):
    coord = discrete_projection_on_ideal(point)
    coord = (16-coord[1], coord[0])
    coord = (coord[0]*(15/16), coord[1]*(25/16))
    return coord

#%% Display
import os
import pickle

template = 'bbox_coord_{}.pickle'
file_name = template.format(1)

os.listdir('coord_player')

plt.figure(figsize=(15, 15))
plt.imshow(img)

file_name = template.format(index)
window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('frame_10.jpg'.format(index), cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [255, 255, 255]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
img = np.copy(dst)

A = np.array([613, 186])
B = np.array([245, 76])

captions = pickle.load(open('coord_player/' + template.format(10), 'rb'))
for capt in captions:
    legend = capt[0]
    proba = capt[1]
    bbox = capt[2]
    center = np.array([right + bbox[0], top + bbox[1] + bbox[3]/2])
    ortho_vec = (center-A) - np.dot(B-A, center-A)*(B-A)/np.linalg.norm(B-A)
    ref_sign = 1-2*int(np.cross(ortho_vec, B-A)>0)
    if ref_sign < 0 and legend==b'person':
        plt.scatter([center[0]], [center[1]], edgecolor='r', color='')
        print(ref_sign)
plt.show()

#%%

import os
import pickle

template = 'bbox_coord_{}.pickle'
file_name = template.format(1)

os.listdir('coord_player')

file_name = template.format(index)
window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('frame_10.jpg'.format(index), cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [255, 255, 255]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
img = np.copy(dst)

A = np.array([613, 186])
B = np.array([245, 76])

point_real = []

captions = pickle.load(open('coord_player/' + template.format(10), 'rb'))
for capt in captions:
    legend = capt[0]
    proba = capt[1]
    bbox = capt[2]
    center = np.array([right + bbox[0], top + bbox[1] + bbox[3]/2])
    ortho_vec = (center-A) - np.dot(B-A, center-A)*(B-A)/np.linalg.norm(B-A)
    ref_sign = 1-2*int(np.cross(ortho_vec, B-A)>0)
    if ref_sign < 0 and legend==b'person':
        point_real.append(center)

plt.figure(figsize=(15, 15))
plt.subplot(121)
for center in point_real:
    plt.scatter([center[0]], [center[1]], edgecolor='r', color='')
plt.imshow(img)
plt.subplot(122)
for center in point_real:
    coord = coord_ideal(center)
    plt.scatter([coord[0]], [coord[1]], edgecolor='r', color='')
plt.show()

#%%

import os
import pickle

template = 'bbox_coord_{}.pickle'
file_name = template.format(1)

os.listdir('coord_player')

file_name = template.format(index)
window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('frame_10.jpg'.format(index), cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [255, 255, 255]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
img = np.copy(dst)

A = np.array([613, 186])
B = np.array([245, 76])

X = []
Y = []

for frame_number in range(1, 11):

    x_real = []
    y_real = []

    captions = pickle.load(open('coord_player/' + template.format(frame_number), 'rb'))
    for capt in captions:
        legend = capt[0]
        proba = capt[1]
        bbox = capt[2]
        center = np.array([right + bbox[0], top + bbox[1] + bbox[3]/2])
        ortho_vec = (center-A) - np.dot(B-A, center-A)*(B-A)/np.linalg.norm(B-A)
        ref_sign = 1-2*int(np.cross(ortho_vec, B-A)>0)
        if ref_sign < 0 and legend==b'person':
            coord = coord_ideal(center)
            x_real.append(coord[0]+np.random.randn())
            y_real.append(coord[1]+np.random.randn())

    X.append(x_real)
    Y.append(y_real)

np.save('x', X)
np.save('y', Y)

#%%