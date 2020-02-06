import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.subplot(111, projection='3d')
pointcloud = np.fromfile(str(r'D:\comVisualStudy\venv\training\label_2\000008.txt'), dtype=np.float32, count=-1).reshape([-1, 4])
x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point
r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
ax.scatter(x, y, z, s=0.1, c='k', marker='.', alpha=0.5) #绘点
# a4 = pointcloud[:, 4]
# a5 = pointcloud[:, 5]
# a6 = pointcloud[:, 6]
# ax.scatter(a4, a5, a6, c='y') #绘点

print(pointcloud.shape)
plt.show()
