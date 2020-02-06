#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/7 16:59
#@Author: 黄怀宇
#@File  : outline-3d.py
#!/usr/bin/env python
from kitti_util import *
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]
    pass


def inverRows(M, r1, r2, r3):
    M[0] = np.multiply(r1, M[0])
    M[1] = np.multiply(r2, M[1])
    M[2] = np.multiply(r3, M[2])
    pass


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    # def rotate matrix in 3d
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    # R = np.array([[1, 0, 0], [0, np.cos(yaw), -np.sin(yaw)], [0, np.sin(yaw), np.cos(yaw)]])  # Rx
    # R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    # R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    # def 8 points' coordinate
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    #  rotate and translate 3D bounding box
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    thita = math.pi/2
    # Rx = np.array([[1, 0, 0], [0, np.cos(thita), -np.sin(thita)], [0, np.sin(thita), np.cos(thita)]])   # Rx
    # Ry = np.array([[np.cos(thita), 0, np.sin(thita)], [0, 1, 0], [-np.sin(thita), 0, np.cos(thita)]])   # Ry
    # Rz = np.array([[np.cos(thita), -np.sin(thita), 0], [np.sin(thita), np.cos(thita), 0], [0, 0, 1]])     # Rz
    # corners_3d_cam2 = np.dot(Ry, corners_3d_cam2)

    return corners_3d_cam2


def draw_box(ax, vertices, axes=[0, 1, 2], color='red'):
    type_c = {'Car': 'b', 'Van': 'g', 'Truck': 'r', 'Pedestrian': 'c', \
              'Person (sitting)': 'm', 'Cyclist': 'y', 'Tram': 'k', 'Misc': 'w'}
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [1, 4], [5, 0]
    ]
    for connections in connections:
        ax.plot(*vertices[:, connections], c=color, lw=1)


def getRequiredFiles3dConcloud(name):
    df = pd.read_csv(r'D:\comVisualStudy\venv\training\label_2\\'+name+'.txt', header=None, sep=' ')  # read labels
    image = cv2.imread(r'D:\comVisualStudy\venv\training\image_2\\'+name+'.png')
    df1 = np.array(pd.read_csv(r'D:\comVisualStudy\venv\training\calib\\'+name+'.txt', header=None, sep=' '))
    pointcloud = np.fromfile(str(r'D:\comVisualStudy\venv\training\velodyne\\'+name+'.bin'), dtype=np.float32,
                             count=-1).reshape([-1, 4])
    print('pointcloud.shape', pointcloud.shape)
    return df, image, df1, pointcloud


def get_P_rect(calib):
    calib = calib[2]
    P2 = np.zeros((3, 4))
    P2[0] = calib[1:5]
    P2[1] = calib[5:9]
    P2[2] = calib[9:13]
    return P2


def get_R_rect(calib):
    calib = calib[4]
    R_rect = np.zeros((3, 3))
    R_rect[0] = calib[1:4]
    R_rect[1] = calib[4:7]
    R_rect[2] = calib[7:10]
    R_rect = np.vstack((np.hstack((R_rect, np.zeros((3, 1)))), [0, 0, 0, 1]))
    return R_rect


# initial setting
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
COLUMN_NAMES = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']


name = '000009'
label, image, calib, pointcloud = getRequiredFiles3dConcloud(name)
calib5 = calib[5]
Tr_velo_to_cam = np.zeros((3, 4))
Tr_velo_to_cam[0] = calib5[1:5]
Tr_velo_to_cam[1] = calib5[5:9]
Tr_velo_to_cam[2] = calib5[9:13]
P_rect = get_P_rect(calib)
R_rect = get_R_rect(calib)
print('P_rect\n', P_rect)
print('R_rect\n', R_rect)
# calib6 = calib[6]
# Tr_imu_to_velo = np.zeros((3, 4))
# Tr_imu_to_velo[0] = calib6[1:5]
# Tr_imu_to_velo[1] = calib6[5:9]
# Tr_imu_to_velo[2] = calib6[9:13]
# rotMatrix, transMatrix = np.hsplit(Tr_velo_to_cam, [3])
T = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
Tinv = np.linalg.inv(T)
# rotMatrix = np.linalg.inv(rotMatrix)

print('R0_rectR0_rectR0_rect\n', Tr_velo_to_cam)
print('T\n', T)
print('Tinv\n', Tinv)

label.columns = COLUMN_NAMES
label.head()
print(label)  # show every labels' meaning ------task
label = label[label.type.isin(['Car', 'Cyclist', 'Truck', 'Van', 'Pedestrian', 'Tram', 'Misc'])]
print('select', label)
# -trick- create and initialize two-dimensional array
parameter = np.zeros((len(label), 7))

dimention_array = 0
while dimention_array < len(label):
    parameter[dimention_array] = np.array(label.loc[dimention_array,
                                                    ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
    dimention_array += 1

# fig = plt.figure(figsize=(7, 13))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dimention_array = 0

color = (255, 255, 0)
lw = 1

while dimention_array < len(label):
    corners_3d_cam2 = compute_3d_box_cam2(*parameter[dimention_array])
    print('坐标系1\n', corners_3d_cam2)
    # swapRows(corners_3d_cam2, 0, 1)
    #
    # # swapRows(corners_3d_cam2, 0, 2)
    # swapRows(corners_3d_cam2, 1, 2)
    # inverRows(corners_3d_cam2, -1, -1, 1)

    # draw_box(ax, (corners_3d_cam2))
    print('corners_3d_cam2\n', corners_3d_cam2)
    corners_3d_cam2 = np.vstack((corners_3d_cam2, [1, 1, 1, 1, 1, 1, 1, 1]))
    print('corners_3d_cam2\n', corners_3d_cam2)
    # print('TTTTTTTTTTTTTT\n', T)

    con3dBox = np.dot(Tinv, corners_3d_cam2)
    # con3dBox = np.dot(corners_3d_cam2.T, np.linalg.inv(T))
    # con3dBox = con3dBox.T
    con3dBox, one = np.vsplit(con3dBox, [3])
    # swapRows(con3dBox, 2, 0)
    draw_box(ax, (con3dBox))

    dimention_array += 1

pnt = pointcloud.T[:, 1::5] # one point in 5 points
ax.view_init(54, -144)
ax.scatter(*pnt, s=0.1, c='k', marker='.', alpha=0.5) #绘点

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim3d(-10,30)
ax.set_ylim3d(-20,20)
ax.set_zlim3d(-2,15)
plt.show()

#  corners_3d_cam2 由label的7个参数计算得出的3*8矩阵
#  extMatrix       由calib的Tr_velo_to_cam生成，旋转、平移矩阵

 # 旋转、平移什么矩阵
 # corners_3d_cam2、点云文件分别对应什么坐标系


