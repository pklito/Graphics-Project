import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# points = [[-0.11, 1.17, 3.1], [-0.11, 1.17, 3.1],
# [-3.76, 0.4, 5.61], [-3.76, 0.4, 5.61],
# [-1.15, -0.82, 6.24], [-1.15, -0.82, 6.24],
# [0.13, -0.66, 5.48], [0.13, -0.66, 5.48],
# [-2.66, -0.28, 5.07], [-2.66, -0.28, 5.07],
# [-1.4, -3.95, 9.99], [-1.4, -3.95, 9.99],
# [0.2, -0.67, 7.01], [0.2, -0.67, 7.01],
# [-2.42, -0.14, 4.38], [-2.42, -0.14, 4.38],
# [2.26, 0.21, 6.11], [2.26, 0.21, 6.11],
# [2.26, 0.16, 6.08], [2.26, 0.16, 6.08],
# [-4.55, 0.6, 6.34], [-4.55, 0.6, 6.34]]

trans = [[[2.27, -0.74, -2.03], [1.93, 1.57, 4.52]], [[1.62, -2.14, 0.84], [1.95, 1.75, 4.54]], [[0.94, -2.12, -1.06], [-2.09, -1.58, 6.42]], [[-1.45, 1.53, 0.59], [1.57, -0.47, 5.23]], [[2.1, 1.57, -0.53], [-0.66, -0.27, 6.76]], [[-2.01, 0.21, -0.18], [-1.29, 0.17, 4.3]], [[0.5, 2.89, -0.89], [0.57, 0.33, 5.94]], [[2.04, 0.31, 1.82], [-2.92, 1.89, 4.51]], [[-1.9, -0.8, 1.24], [1.55, -0.46, 5.31]], [[2.08, 1.64, -0.65], [-2.81, 1.93, 4.21]], [[-0.33, -2.87, 1.02], [-3.32, 0.41, 5.76]], [[0.45, 2.78, -0.94], [1.47, -0.39, 4.91]], [[-1.46, 0.99, -1.47], [-4.12, 0.17, 7.12]], [[2.01, 1.57, -0.54], [-2.93, -1.2, 5.9]], [[2.72, -0.0, -0.98], [-3.03, 0.08, 6.95]], [[-1.39, 1.1, -1.73], [-1.42, 0.19, 4.8]]]
# matrix, _ = cv.Rodrigues(np.array(rvec))

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
# points = [[p[0] - points[0][0], p[1]- points[0][1], p[2] - points[0][2]] for p in points]
# points = [[p[0] + 0.35*p[2], p[1], p[2]] for p in points]
points = [t[1] for t in trans] + [[0,0,0]]

arrows = False
if arrows:
    for i in range(0, len(points) - 1, 2):
        ax.quiver(points[i][0], points[i][1], points[i][2],
                points[i+1][0] - points[i][0], points[i+1][1] - points[i][1], points[i+1][2] - points[i][2],
                arrow_length_ratio=0.1, color='b')

x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]

ax.scatter(x, y, z, c='r', marker='o')
# ax.set_xlim([-5, 2])
# ax.set_ylim([-5, 2])
# ax.set_zlim([-5, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()