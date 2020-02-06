import numpy as np
import os.path as osp
import json
def load_data_from_nuscence(scene, index):
    """
    从nuscence中读取数据. 要求速度尽可能快, 所有坐标以lidar坐标系为准，即需要把label中的gt转化到lidar下
    :param scene: (string) scene场景的名称 e.g.'scene-0061'
    :param index: (int) 该scene场景下的第index个frame序号.
    :return:
        gt_dict: {'gt_boxes':(numpy.array)[num, 7], 'gt_names':(string)[num]}标签 num:label中object个数 7:x,y,z,w,l,h,rotation. 注意：xyz需要坐标变换过
        calib: {'P_cam_front'(cam_front内参矩阵), 'Tr_cam_front_to_lidar'(cam_front到lidar的坐标变换矩阵), 'Gt_to_lidar'(label坐标到lidar的变换矩阵}
        points: (numpy.float32) lidar点云 shape:[N, 4] 点云, 4:x,y,z,radius
        image_f: (numpy.uint8) 前方相机图像
    """

    with open(osp.join('D:\\v1.0-mini\\v1.0-mini', '{}.json'.format('scene'))) as f:
        table = json.load(f)
    return gt_dict, calib, points, image_f

with open(osp.join('D:\\v1.0-mini\\v1.0-mini', '{}.json'.format('scene'))) as f:
        table = json.load(f)
print(table)
