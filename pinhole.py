#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/7 16:59
#@Author: 黄怀宇
#@File  : outline-3d.py
#!/usr/bin/env python
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # grey in pycharm but cannot be ommitted


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    # def rotate matrix in 3d
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    # def 8 points' coordinate
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    #  rotate and translate 3D bounding box
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def draw_box(ax, vertices, axes=[0, 1, 2], color='red'):
    vertices = vertices[axes, :]
    connections = [[0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]]
    for connections in connections:
        print('*vertices[:, connections]', *vertices[:, connections])
        ax.plot(*vertices[:, connections], c=color, lw=1)


def draw_box_pinhole(img, pts_2D, axes=[0, 1], color=(255, 255, 0), lw=1):
    pts_2D = pts_2D[axes, :]
    connections = [[0, 1], [1, 2], [2, 3], [3, 0],
                   [4, 5], [5, 6], [6, 7], [7, 4],
                   [0, 4], [1, 5], [2, 6], [3, 7],
                   [1, 4], [5, 0]]
    for connections in connections:
        cv2.line(img, (*pts_2D[:, np.array([connections[0]])][1], *pts_2D[:, np.array([connections[0]])][0]),
                 (*pts_2D[:, np.array([connections[1]])][1], *pts_2D[:, np.array([connections[1]])][0]), color, lw)


def getRequiredFilesPinhole(name):
    df = pd.read_csv(r'D:\comVisualStudy\venv\training\label_2\\'+name+'.txt', header=None, sep=' ')  # read labels
    image = cv2.imread(r'D:\comVisualStudy\venv\training\image_2\\'+name+'.png')
    df1 = np.array(pd.read_csv(r'D:\comVisualStudy\venv\training\calib\\'+name+'.txt', header=None, sep=' '))
    return df, image, df1


def getIntrinsicParameters(calib):
    calib = calib[2]
    P2 = np.zeros((3, 4))
    P2[0] = calib[1:5]
    P2[1] = calib[5:9]
    P2[2] = calib[9:13]
    fx = P2[0][0]
    fy = P2[1][1]
    cu = P2[0][2]
    cv = P2[1][2]
    return fx, fy, cu, cv


def get3dParameterMatrix(label):
    dimention_array = 0
    parameter = np.zeros((len(label), 7))    # -trick- create and initialize two-dimensional array
    while dimention_array < len(label):
        parameter[dimention_array] = np.array(label.loc[dimention_array,
                                                        ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        dimention_array += 1
    return parameter


def cal3dProjectTo2d(corners_3d_cam2, fx, fy, cu, cv):
    pts_2D = np.zeros((2, 8))
    i = 0
    while i < 8:
        pts_2D[0][i] = fy * corners_3d_cam2[1][i] / corners_3d_cam2[2][i] + cv
        pts_2D[1][i] = fx * corners_3d_cam2[0][i] / corners_3d_cam2[2][i] + cu
        i += 1
    return pts_2D


COLUMN_NAMES = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
# print omitted labels
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# lack argv input
name = '000008'

label, image, calib = getRequiredFilesPinhole(name)
fx, fy, cu, cv = getIntrinsicParameters(calib)

label.columns = COLUMN_NAMES
label.head()  # .head('n') show The first n lines
print(label)  # show every labels' meaning
label = label[label.type.isin(['Car', 'Cyclist', 'Truck', 'Van', 'Pedestrian', 'Tram', 'Misc'])]
print('select', label)
parameter = get3dParameterMatrix(label)

# draw 2d project circulatory according to 3d parameter
dimention_array = 0
while dimention_array < len(label):
    corners_3d_cam2 = compute_3d_box_cam2(*parameter[dimention_array])
    pts_2D = cal3dProjectTo2d(corners_3d_cam2, fx, fy, cu, cv).astype(np.int16)
    draw_box_pinhole(image, pts_2D)
    dimention_array += 1

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


