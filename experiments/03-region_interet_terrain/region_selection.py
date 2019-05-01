#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:32:15 2019
Selection d'une région d'intérêt convexe:
    - Par séparation successive d'hyperplans
    - Création d'un masque de départ

@author: tdesfont
"""

#%% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

#%% Read images
path_image = '../../frames/'
image_name = 'Angle1_frame_1.jpg'
# read image
img = cv2.imread(path_image + image_name, 0)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
#%%

nodes = [[ 37,  69],
         [193,  40],
         [569, 119],
         [493, 359],
         [114, 356]]

nodes.insert(0, nodes[-1])
Nodes = np.array(nodes)

plt.figure(figsize=(15, 5))
plt.imshow(img)
plt.plot(Nodes[:, 0], Nodes[:, 1], 'r-')
plt.show()

#%%
# Calcul trop long à faire de manière matricielle sur C...

if not 'mask.npy' in os.listdir():
    mask = np.ones(img.shape)

    for i in range(4):
        print('Step i {}'.format(i))
        A = Nodes[i]
        B = Nodes[i+1]
        C = Nodes[i+2]

        ortho_ref_vec = (C-A) - np.dot(B-A, C-A)*(B-A)/np.linalg.norm(B-A)
        ref_sign = 1-2*int(np.cross(ortho_ref_vec, B-A)>0)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                X = np.array([j, i])
                ortho_vec = (X-A) - np.dot(B-A, X-A)*(B-A)/np.linalg.norm(B-A)
                sign = 1-2*int(np.cross(ortho_vec, B-A)*ref_sign>0)
                if sign == -1:
                    mask[i, j] = 0
        np.save('mask', mask)
else:
    mask = np.load('mask.npy')

#%% Display mask

plt.figure()
plt.imshow(img)
plt.imshow(mask, alpha=0.5)
plt.colorbar()
plt.show()

#%% Read a video with a mask

roi = cv2.bitwise_and(img, img, mask = mask.astype('int8'))

plt.figure(figsize=(10, 5))
plt.imshow(roi)
plt.savefig('ROI')
plt.show()

edges = cv2.Canny(roi, 50, 200)

plt.figure(figsize=(10, 5))
plt.imshow(edges, cmap = 'magma', alpha=0.9)
plt.savefig('edgesOnROI')


