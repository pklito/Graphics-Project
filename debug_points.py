import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = [
[3.38, -1.4, 4.69], [3.39, -1.83, 4.43],
[-0.42, 1.49, 4.01], [-0.51, 1.16, 3.65],
[2.0, -0.86, 3.75], [2.0, -1.28, 3.49],
[1.24, -0.58, 3.33], [1.25, -1.01, 3.07],
[1.91, -1.0, 3.72], [2.21, -0.7, 3.45],
[1.11, -0.66, 3.2], [1.41, -0.37, 2.92],
[-0.95, 1.02, 5.04], [-1.02, 0.65, 4.71],
[0.33, 1.19, 4.6], [0.34, 0.84, 4.25],
[2.79, -1.19, 4.35], [2.81, -1.62, 4.1],
[-0.16, 0.7, 5.53], [-0.18, 0.31, 5.21],
[1.66, -2.38, 7.53], [1.66, -2.76, 7.2],
[-1.52, 0.49, 6.19], [-1.56, 0.12, 5.85],
[1.69, -1.52, 5.07], [1.69, -1.95, 4.8]
]

rvecs = [[-0.46, 1.24, 2.21], [2.31, -0.65, 0.01], [2.04, -0.69, 0.38], [-0.44, 1.24, 2.24], [-2.02, -1.76, -1.48], [0.13, 2.31, 0.9], [-0.67, -0.76, -1.94], [2.29, -0.65, 0.3], [2.0, -0.75, 0.48], [-0.79, 0.5, 0.96], [-0.55, -0.84, -1.85], [2.23, -0.55, 0.14], [-0.59, -1.08, -1.95]]
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

points = points[0::2]
points = [(matrix.T @ np.array(point)).flatten() for point in points]

points = [[p[0] - points[0][0], p[1]- points[0][1], p[2] - points[0][2]] for p in points]
points = [[p[0] + 0.35*p[2], p[1], p[2]] for p in points]


x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()