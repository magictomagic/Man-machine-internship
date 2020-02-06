#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/4 19:26
#@Author: 黄怀宇
#@File  : funLib.py
import numpy as np


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    return: 3xn in cam2 coordinate
    """
    # def rotate matrix in 3d
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    # def 8 points' coordinate
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # last get coordinate
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for connections in connections:
        ax.plot(*vertices[:, connections], c=color, lw=0.5)