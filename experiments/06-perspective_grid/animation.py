#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:18:44 2019
@author: tdesfont
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation

class SubplotAnimation(animation.TimedAnimation):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlim(-5, 20)
        ax1.set_ylim(-5, 30)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        self.line_players = Line2D([], [], marker='o', markeredgecolor='k',
                        alpha=0.8, markersize=10, linewidth=0, color='r')
        self.line_path = Line2D([], [], marker='o', markeredgecolor='k',
                        alpha=0.5, markersize=5, linewidth=0, color='b')
        ax1.add_line(self.line_players)
        ax1.add_line(self.line_path)

        animation.TimedAnimation.__init__(self, fig, interval=1000, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        self.line_players.set_data(X[i], Y[i])
        if i > 0:
            self.line_path.set_data(X[i-1], Y[i-1])
        self._drawn_artists = [self.line_players, self.line_path]

    def new_frame_seq(self):
        return iter(range(len(self.X)))

    def __init__draw(self):
        lines = [self.line_players, self.line_path]
        for l in lines:
            l.set_data([], [])

X = np.load('x.npy')
Y = np.load('y.npy')
ani = SubplotAnimation(X, Y)
ani.save('multiple_cells.gif', dpi=80, writer='imagemagick')
plt.show()