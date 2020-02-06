import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def draw_box(ax, vertices, axes=[0, 1, 2], color='yellow'):
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for connections in connections:
        ax.plot(*vertices[:, connections], c=color, lw=2)


# print omitted labels
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# set labels
COLUMN_NAMES = ['type', 'truncated', 'occluded', 'alpha',

                # 2d parameter
                'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',   # 4 * bbox

                # 3d parameter
                'height', 'width', 'length',                            # 3 * dimensions
                'pos_x', 'pos_y', 'pos_z',                              # 3 * locations: in camera
                'rot_y']
# read labels
df = pd.read_csv(r'D:\comVisualStudy\venv\000002.txt', header=None, sep=' ')
df.columns = COLUMN_NAMES
df.head()
box = np.zeros((len(df), 7))
parameter = np.array(df.loc[0, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])

corners_3d_cam2 = compute_3d_box_cam2(*parameter)
print(corners_3d_cam2.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.view_init(40, 150)
draw_box(ax, (corners_3d_cam2))
plt.show()
