#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/17 2:32
#@Author: 黄怀宇
#@File  : codeRepetition.py
# from sklearn.metrics import accuracy_score #works
# print(accuracy_score([1, 1, 0], [1, 0, 1]))

from nuscenes.nuscenes import NuScenes
import numpy as np


nusc = NuScenes(version='v1.0-mini', dataroot='D:\\v1.0-mini', verbose=True)
print('all scene')
nusc.list_scenes()      # 列出所有的场景

my_scene = nusc.scene[0]
print('\nmy_scene\n', my_scene)         # 列出所有场景中的第一个场景对应的所有 token

first_sample_token = my_scene['first_sample_token']     # scene.json 中第一个元素组中的 first_sample_token

# sample.json 中第一帧的所有元素(token、timestamp、prev、next、scene_token) 和
# 汽车所有传感器对应的数据

my_sample = nusc.get('sample', first_sample_token)
print('my_sample\n', my_sample)
#       只取汽车所有传感器对应的数据的方法 print('my_sample\n', my_sample['data'])
#       只取CAM_FRONT 的方法
# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# print(cam_front_data)
#               以此类推
#       获取传感器对应的标注框的方法(以前一个例子为例)[可视化]
# nusc.render_sample_data(cam_front_data['token'])

nusc.list_sample(my_sample['token'])      # 列出了与示例相关的所有sample_data关键帧和sample_annotation

my_annotation_token = my_sample['anns'][18]  # 取第19个 sample_annotation_token（其它的好像没标注好）
my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)     # 得到第一个的详细注释 sample_annotation.json 的一个元素组
print('\nmy_annotation_metadata\n', my_annotation_metadata)
# nusc.render_annotation(my_annotation_token)     # [可视化]
my_instance = nusc.instance[599]        # instance.json 的 4196 行开始 ？
print('my_instance\n', my_instance)
