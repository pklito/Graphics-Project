import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = [
[0.71, -0.55, 5.15], [0.47, -0.4, 4.74],
[1.04, 0.89, 3.8], [0.81, 1.15, 3.45],
[-3.48, -0.25, 7.9], [-3.14, -0.19, 7.54],
[2.86, -1.32, 7.2], [2.56, -1.39, 6.8],
[-3.41, -0.31, 7.36], [-3.68, -0.08, 7.01],
[-3.57, -1.16, 7.13], [-3.41, -0.69, 7.08]]
rvecs = [[-0.11, 0.64, -1.37], [0.13, 0.84, -1.28], [-1.36, 2.0, -0.38], [2.06, -1.53, -0.57], [3.52, 0.09, -1.07], [0.93, -1.23, 0.96]]
randindex = np.random.randint(len(rvecs))
rvec = rvecs[4]
print(4)



matrix, _ = cv.Rodrigues(np.array(rvec))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta = np.radians(0)
rotation_matrix_y = np.array([
    
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta),0, np.cos(theta)]
])

# points = points[0::2]
# points = [(matrix.T @ np.array(point)).flatten() for point in points]
points = [[p[0] - points[0][0], p[1]- points[0][1], p[2] - points[0][2]] for p in points]
points = [[p[0] + 0.35*p[2], p[1], p[2]] for p in points]


for i in range(0, len(points) - 1, 2):
    ax.quiver(points[i][0], points[i][1], points[i][2],
              points[i+1][0] - points[i][0], points[i+1][1] - points[i][1], points[i+1][2] - points[i][2],
              arrow_length_ratio=0.1, color='b')

x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlim([-5, 2])
ax.set_ylim([-5, 2])
ax.set_zlim([-5, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()