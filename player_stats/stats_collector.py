#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:31:23 2019
@author: tdesfont

    Collect and treat stats.

"""

#%% Imports
import os
import pickle
import matplotlib.pyplot as plt
#%%

folder = './shared/player_stats/coord_bbox/'
for i in range(1, 20):
    filename = 'bbox_coord_{}.pickle'.format(i)

file = open(folder + filename, 'rb')
object_file = pickle.load(file)

x_ = []
y_ = []
for i, _ in enumerate(object_file):
    x, y, width, height = object_file[i][-1]
    x_.append(x+width/2)
    y_.append(y)

plt.figure(figsize=(10, 10))
plt.plot(x_, y_, 'ro')
plt.show()