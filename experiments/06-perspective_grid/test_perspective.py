#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Jun  4 21:03:28 2019
    @author: tdesfont
"""

# Imports
import sys
from random import randint
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Try to find the 4th corner
window_name = "padded"
borderType = cv.BORDER_CONSTANT
src = cv.imread('sample_image.jpg', cv.IMREAD_COLOR)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

top = int(0.05 * src.shape[0])  # shape[0] = rows
bottom = int(0.8 * src.shape[0])
left = int(0.05 * src.shape[1])  # shape[1] = cols
right = left

value = [0, 0, 0]
dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
cv.imshow(window_name, dst)

# Approximately set it to [346, 640]
points = np.array([[69,   89],
                   [222,  62],
                   [614, 132],
                   [340, 640]])

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

class parallelogram:

    def __init__(self, points):
        """
            ordered points
        """
        self.points = points
        self.center = line(self.points[0], points[2])*line(self.points[1], points[3])
        global img
        img = cv.circle(img, (int(self.center[0]), int(self.center[1])), 3, (255,0,0), -1)
        for point in self.points:
            img = cv.circle(img, (int(point[0]), int(point[1])), 3, (255,0,0), -1)

    def divide(self):
        #
        Ia = line(self.points[0], self.points[3])*line(self.points[1], self.points[2])
        A_ = line(Ia, self.center)*line(self.points[0], self.points[1])
        C_ = line(Ia, self.center)*line(self.points[2], self.points[3])
        #
        Ib = line(self.points[0], self.points[1])*line(self.points[2], self.points[3])
        B_ = line(Ib, self.center)*line(self.points[1], self.points[2])
        D_ = line(Ib, self.center)*line(self.points[0], self.points[3])
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

recursive_ortho_grid(4, points)

cv.imshow(window_name, img)