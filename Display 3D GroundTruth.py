#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/8 1:57
#@Author: 黄怀宇
#@File  : Display 3D GroundTruth.py
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import glob
from src import parseTrackletXML as pt_XML
from kitti_foundation import Kitti, Kitti_util
image_type = 'color'  # 'gray' or 'color' image
mode = '00' if image_type == 'gray' else '02'  # image_00 = 'graye image' , image_02 = 'color image'

image_path = 'image_' + mode + '/data'
velo_path = './velodyne_points/data'
xml_path = "./tracklet_labels.xml"
v2c_filepath = './calib_velo_to_cam.txt'
c2c_filepath = './calib_cam_to_cam.txt'
frame = 89

check = Kitti_util(frame=frame, velo_path=velo_path, camera_path=image_path, \
                   xml_path=xml_path, v2c_path=v2c_filepath, c2c_path=c2c_filepath)

# bring velo points & tracklet info
points = check.velo_file
tracklet_, type_ = check.tracklet_info

print(points.shape)
print('The number of GT : ', len(tracklet_[frame]))


def draw_3d_box(tracklet_, type_):
    """ draw 3d bounding box """

    type_c = {'Car': 'b', 'Van': 'g', 'Truck': 'r', 'Pedestrian': 'c', \
              'Person (sitting)': 'm', 'Cyclist': 'y', 'Tram': 'k', 'Misc': 'w'}

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], \
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    for i, j in zip(tracklet_[frame], type_[frame]):
        for k in line_order:
            ax.plot3D(*zip(i.T[k[1]], i.T[k[0]]), lw=1.5, color=type_c[j])


frame = 89
image = check.camera_file
tracklet_, type_ = check.tracklet_info

tracklet2d = []
for i, j in zip(tracklet_[frame], type_[frame]):
    point = i.T
    chk,_ = check._Kitti_util__velo_2_img_projection(point)
    tracklet2d.append(chk)

type_c = { 'Car': (0, 0, 255), 'Van': (0, 255, 0), 'Truck': (255, 0, 0), 'Pedestrian': (0,255,255), \
      'Person (sitting)': (255, 0, 255), 'Cyclist': (255, 255, 0), 'Tram': (0, 0, 0), 'Misc': (255, 255, 255)}

line_order = ([0, 1], [1, 2],[2, 3],[3, 0], [4, 5], [5, 6], \
         [6 ,7], [7, 4], [4, 0], [5, 1], [6 ,2], [7, 3])

for i, j in zip(tracklet2d, type_[frame]):
    for k in line_order:
        cv2.line(image, (int(i[0][k[0]]), int(i[1][k[0]])), (int(i[0][k[1]]), int(i[1][k[1]])), type_c[j], 2)

plt.subplots(1,1, figsize = (12,4))
plt.title("3D Tracklet display on image")
plt.axis('off')
plt.imshow(image)