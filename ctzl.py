#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/8 3:32
#@Author: 黄怀宇
#@File  : ctzl.py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
frame = 89
def draw_3d_box(tracklet_, type_):
    """ draw 3d bounding box """

    type_c = {'Car': 'b', 'Van': 'g', 'Truck': 'r', 'Pedestrian': 'c', \
              'Person (sitting)': 'm', 'Cyclist': 'y', 'Tram': 'k', 'Misc': 'w'}

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], \
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    for i, j in zip(tracklet_[frame], type_[frame]):
        for k in line_order:
            ax.plot3D(*zip(i.T[k[1]], i.T[k[0]]), lw=1.5, color=type_c[j])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pointcloud = np.fromfile(str("000001.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point
r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
ax.scatter(x, y, z, s=0.1, c='k', marker='.', alpha=0.5) #绘点
draw_3d_box()
plt.show()

# def plot_linear_cube(a0, a1, a2, a3, a4, a5, a6, a7, color='red'):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     xx = [a0, a1, a2, a3]
#     yy = [a4, a5, a6, a7]
#     kwargs = {'alpha': 1, 'color': color}
#     ax.plot3D(xx, yy, [z]*5, **kwargs)
#     # ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
#     # ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
#     # ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
#     # ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
#     # ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
#     plt.title('Cube')
#     plt.show()
#
# plot_linear_cube(10, 20, 30, 40, 50, 60)
