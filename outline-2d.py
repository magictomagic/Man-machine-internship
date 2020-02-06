#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/6 18:25
#@Author: 黄怀宇
#@File  : outline-2d.py
import pandas as pd
import cv2
import numpy as np
# print omitted labels
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# set labels
COLUMN_NAMES = ['type', 'truncated', 'occluded', 'alpha',
                'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',   # 4 * bbox
                'height', 'width', 'length',                            # 3 * dimensions
                'pos_x', 'pos_y', 'pos_z',                              # 3 * locations
                'rot_y']

# read labels
df = pd.read_csv(r'D:\comVisualStudy\venv\training\label_2\000008.txt', header=None, sep=' ')

# read & save pictures
image = cv2.imread(r'D:\comVisualStudy\venv\training\image_2\000008.png')

# corresponding labels
df.columns = COLUMN_NAMES
df.head()

# show every labels' meaning ------task
print(df)

# -trick- create and initialize two-dimensional array
box = np.zeros((len(df), 4))

# extract appointed labels
dimention_array = 0
while dimention_array < len(df):
    box[dimention_array] = np.array(df.loc[dimention_array, ['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
    dimention_array += 1
print(box)

# draw rectangles
dimention_array = 0
while dimention_array < len(df):
    top_left = int(box[dimention_array][0]), int(box[dimention_array][1])
    bottom_right = int(box[dimention_array][2]), int(box[dimention_array][3])
    # cv2.rectangle()
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), 2)      # last parameter: lines' width
    dimention_array += 1

# show images
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
