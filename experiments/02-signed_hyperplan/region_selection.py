#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:32:15 2019
Selection d'une région d'intérêt convexe:
    -

@author: tdesfont
"""

#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%

n = 100
img = np.ones((n, n))

A = np.array([10, 20])
B = np.array([30, 10])
C = np.array([10, 5])

plt.plot([10], [20], 'ro')
plt.plot([30], [10], 'ro')
plt.plot([10], [5], 'ro')

ortho_ref_vec = (C-A) - np.dot(B-A, C-A)*(B-A)/np.linalg.norm(B-A)
ref_sign = 1-2*int(np.cross(ortho_ref_vec, B-A)>0)

for i in range(n):
    for j in range(n):
        X = np.array([j, i])
        ortho_vec = (X-A) - np.dot(B-A, X-A)*(B-A)/np.linalg.norm(B-A)
        img[i, j] = 1-2*int(np.cross(ortho_vec, B-A)*ref_sign>0)

plt.imshow(img)
plt.colorbar()
plt.show()

#%%
